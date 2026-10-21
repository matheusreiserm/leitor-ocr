import sys, time, easyocr
t = time.time(); r = easyocr.Reader(['es', 'pt'], gpu=False)
print(f'== easyocr ({time.time()-t:.0f}s)\n' + '\n'.join(r.readtext(sys.argv[1], detail=0, paragraph=True))[:500])
