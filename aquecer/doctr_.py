import sys, time
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
doc = DocumentFile.from_images([sys.argv[1]])
for reco in ('crnn_vgg16_bn', 'parseq'):
    t = time.time(); m = ocr_predictor(det_arch='fast_base', reco_arch=reco, pretrained=True)
    txt = m(doc).render(); print(f'== doctr {reco} ({time.time()-t:.0f}s)\n{txt[:500]}')
