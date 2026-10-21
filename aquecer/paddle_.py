import sys, time
from paddleocr import PaddleOCR
for lang in ('es', 'pt'):
    t = time.time()
    o = PaddleOCR(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False)
    r = o.predict(sys.argv[1])[0]
    print(f'== paddle {lang} ({time.time()-t:.0f}s)\n' + ' '.join(r['rec_texts'])[:500])
