import cv2, numpy as np, sys
from pathlib import Path
ESC = 2
def textura(h, w, seed, lado):
    rng = np.random.default_rng(seed)
    base = np.array([232, 243, 249], np.float32)  # creme (BGR)
    tex = np.ones((h, w, 3), np.float32) * base
    # manchas de baixa frequência
    for sig, amp in ((h / 6, 2.4), (h / 25, 1.3)):
        n = rng.standard_normal((max(4, int(h / sig * 4)), max(4, int(w / sig * 4)))).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC); n = cv2.GaussianBlur(n, (0, 0), sig / 6)
        n /= (n.std() + 1e-6); tex += (n * amp)[..., None]
    # grão fino
    g = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.8); g /= g.std()
    tex += (g * 2.2)[..., None]
    # fibras
    fib = np.zeros((h, w), np.float32)
    for _ in range(int(h * w / 9000)):
        x, y = rng.integers(0, w), rng.integers(0, h); L = rng.integers(6, 30); a = rng.uniform(0, np.pi)
        cv2.line(fib, (int(x), int(y)), (int(x + L * np.cos(a)), int(y + L * np.sin(a))), float(rng.uniform(0.3, 1)), 1, cv2.LINE_AA)
    fib = cv2.GaussianBlur(fib, (0, 0), 0.7); tex -= (fib * 7)[..., None]
    # vinheta suave + sombra da lombada
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]) / (0.06 * w)
    tex *= (0.965 + 0.035 * np.clip(d, 0, 1))[..., None]
    if lado:
        dx = (w - 1 - xx) if lado == 'esq' else xx
        tex *= (1 - 0.10 * np.exp(-dx / (0.035 * w)) - 0.035 * np.exp(-dx / (0.12 * w)))[..., None]
    return np.clip(tex, 0, 255)
def aplicar(pg, seed, lado, capa=False):
    big = cv2.resize(pg, None, fx=ESC, fy=ESC, interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    if capa: return np.clip(big, 0, 255).astype(np.uint8)
    t = textura(big.shape[0], big.shape[1], seed, lado)
    return np.clip(big / 255.0 * t, 0, 255).astype(np.uint8)  # multiplicar: o conteúdo não muda, só o "papel"
if __name__ == '__main__':
    src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    only = sys.argv[3:] 
    for k, f in enumerate(sorted(src.glob('*.png'))):
        if only and f.name not in only: continue
        n = k + 1
        lado = None if n == 1 else ('esq' if n % 2 == 0 else 'dir')
        r = aplicar(cv2.imread(str(f)), 1000 + n, lado, capa=(n == 1))
        cv2.imwrite(str(out / (f.stem + '.jpg')), r, [cv2.IMWRITE_JPEG_QUALITY, 93, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
