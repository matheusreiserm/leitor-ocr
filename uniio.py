"""cv2.imread/imwrite que funcionam com caminhos Unicode no Windows (o OpenCV falha calado com acentos)."""
import cv2, numpy as np
from pathlib import Path
_r, _w = cv2.imread, cv2.imwrite
def imread(p, flags=cv2.IMREAD_COLOR):
    try: return cv2.imdecode(np.fromfile(str(p), np.uint8), flags)
    except (FileNotFoundError, OSError): return None
def imwrite(p, img, params=()):
    ok, buf = cv2.imencode(Path(p).suffix or '.png', img, list(params))
    if ok: buf.tofile(str(p))
    return ok
cv2.imread, cv2.imwrite = imread, imwrite
