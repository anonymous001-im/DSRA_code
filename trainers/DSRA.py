import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name == "ViT-L/14":
        url = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
    else:
        try:
            url = clip._MODELS[backbone_name]
        except KeyError:
            url = backbone_name

    print(f"Downloading CLIP model from: {url}")
    try:
        model_path = clip._download(url)
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise e

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DSRA.N_CTX
        ctx_init = cfg.TRAINER.DSRA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]


        self.n_prompts = 3


        cfg.defrost()
        cfg.TRAINER.DSRA.CSC = False
        cfg.freeze()


        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors_base = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f"Mode: Ensemble (3 Prompts) + Shared Context. Init: {prompt_prefix}")

        self.ctx_list = nn.ParameterList()
        for _ in range(self.n_prompts):
            if ctx_init:
                ctx_vectors = ctx_vectors_base.clone()
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)

            self.ctx_list.append(nn.Parameter(ctx_vectors))

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg.TRAINER.DSRA.CLASS_TOKEN_POSITION

    def forward(self, prompt_idx=0):
        ctx = self.ctx_list[prompt_idx]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [prefix, ctx, suffix],
                dim=1,
            )
        else:
             raise ValueError("Please use class_token_position='end'")
        return prompts


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()


        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = x + self.scale * self.fc(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        feature_dim = clip_model.visual.output_dim
        self.adapter = Adapter(feature_dim, reduction=4).type(self.dtype)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = self.adapter(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits_list = []
        text_features_list = []


        for i in range(self.prompt_learner.n_prompts):
            prompts = self.prompt_learner(prompt_idx=i)
            text_features = self.text_encoder(prompts, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            text_features_list.append(text_features)
            logits = logit_scale * image_features @ text_features.t()
            logits_list.append(logits)

        return logits_list, text_features_list
    def encode_image_features(self, image, return_raw=False):
        image = image.type(self.dtype)

        raw_features = self.image_encoder(image)
        adapted_features = self.adapter(raw_features)
        adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)

        if return_raw:
            raw_features = raw_features / raw_features.norm(dim=-1, keepdim=True)
            return raw_features, adapted_features

        return adapted_features


@TRAINER_REGISTRY.register()
class DSRA(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.DSRA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.DSRA.PREC == "fp32" or cfg.TRAINER.DSRA.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP (Ensemble + Adapter + ViT-L Version)")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "adapter" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = torch.optim.SGD(
            trainable_params,
            lr=cfg.OPTIM.LR,
            momentum=0.9,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY
        )
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.DSRA.PREC == "amp" else None
        device_count = torch.cuda.device_count()
        if device_count > 1:
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, *_ = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.DSRA.PREC
        if prec == "amp":
            with autocast():
                logits_list, text_features_list = self.model(image)


                loss_ce = 0
                for logits in logits_list:
                    loss_ce += F.cross_entropy(logits, label)
                loss_ce = loss_ce / len(logits_list)


                p1, p2, p3 = text_features_list
                sim12 = F.cosine_similarity(p1, p2, dim=1).abs().mean()
                sim23 = F.cosine_similarity(p2, p3, dim=1).abs().mean()
                sim13 = F.cosine_similarity(p1, p3, dim=1).abs().mean()
                loss_div = (sim12 + sim23 + sim13) / 3.0

                loss = loss_ce + 0.1 * loss_div

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits_list, text_features_list = self.model(image)
            loss_ce = 0
            for logits in logits_list:
                loss_ce += F.cross_entropy(logits, label)
            loss_ce /= len(logits_list)

            p1, p2, p3 = text_features_list
            sim12 = F.cosine_similarity(p1, p2, dim=1).abs().mean()
            sim23 = F.cosine_similarity(p2, p3, dim=1).abs().mean()
            sim13 = F.cosine_similarity(p1, p3, dim=1).abs().mean()
            loss_div = (sim12 + sim23 + sim13) / 3.0

            loss = loss_ce + 0.1 * loss_div
            self.model_backward_and_update(loss)

        with torch.no_grad():
            avg_logits = torch.stack(logits_list).mean(dim=0)
            acc = compute_accuracy(avg_logits, label)[0].item()

        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
            "div": loss_div.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        logits_list, _ = self.model(input)
        logits_orig = torch.stack(logits_list).mean(dim=0)


        return logits_orig


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory: return
        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None: model_file = "model.pth.tar-" + str(epoch)
        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path): raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            if "token_prefix" in state_dict: del state_dict["token_prefix"]
            if "token_suffix" in state_dict: del state_dict["token_suffix"]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
