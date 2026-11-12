import glob, os
def load_librispeech(root, limit=None):
    items = []
    for trans in sorted(glob.glob(f"{root}/**/*.trans.txt", recursive=True)):
        d = os.path.dirname(trans)
        for line in open(trans):
            uid, text = line.strip().split(" ", 1)
            items.append((os.path.join(d, uid + ".flac"), text))
    return items[:limit] if limit else items
