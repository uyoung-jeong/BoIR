from yacs.config import CfgNode

def get_cfg() -> CfgNode:
    from .default import _C

    return _C.clone()

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg