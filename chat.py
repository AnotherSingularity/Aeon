import argparse, os, yaml, torch
from transformers import AutoTokenizer
from aeon.hybrid import HybridModel
from aeon.transformer import config_from_pretrained

def load_model(ckpt_path, config_path, device):
    cfg = yaml.safe_load(open(config_path))
    mcfg = cfg['model']
    r1_dir = os.environ['AEON_R1_DIR']
    qcfg = config_from_pretrained(r1_dir)
    model = HybridModel(
        h_rec=mcfg['h_rec'], K=mcfg['K'],
        transformer_config=qcfg, substrate=mcfg['substrate'],
        freeze_backbone=True,
        use_embedding_input=mcfg.get('use_embedding_input', True),
        dtype=torch.bfloat16,
    ).to(device)
    model.to(dtype=torch.bfloat16)
    model.transformer.gamma.data = model.transformer.gamma.data.float()
    model.recursion.float()
    model.transformer.load_pretrained(r1_dir)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    info = model.load_state_dict(blob['model'], strict=False)
    print(f"[load] step={blob['step']}, gamma={model.transformer.gamma.item():.6f}")
    model.eval()
    tok = AutoTokenizer.from_pretrained(r1_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok

@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=150, temperature=0.7, device='cuda'):
    ids = tok(prompt, return_tensors='pt').input_ids.to(device)
    out = ids.clone()
    for _ in range(max_new_tokens):
        result = model(input_ids=out)
        logits = result.logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=-1)
        if next_id.item() == tok.eos_token_id:
            break
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/stage1_hybrid/ckpt_2000.pt')
    ap.add_argument('--config', default='configs/stage1_hybrid.yaml')
    ap.add_argument('--max_new_tokens', type=int, default=150)
    ap.add_argument('--temperature', type=float, default=0.7)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[init] device={device}")
    model, tok = load_model(args.ckpt, args.config, device)
    print(f"[init] ready. type prompt, empty line + enter to submit, ctrl-c to exit\n")
    while True:
        try:
            print("> ", end='', flush=True)
            lines = []
            while True:
                line = input()
                if line == '':
                    break
                lines.append(line)
            prompt = '\n'.join(lines)
            if not prompt.strip():
                continue
            print()
            out = generate(model, tok, prompt, args.max_new_tokens, args.temperature, device)
            print(out[len(prompt):] if out.startswith(prompt) else out)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n[exit]")
            break

if __name__ == '__main__':
    main()
