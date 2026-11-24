"""Huellas visuales multivista y componentes de imágenes relacionadas."""
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.fft import dctn


def perceptual_hash(pixels):
    coefficients = dctn(pixels, norm="ortho")[:8, :8].ravel()[1:]
    bits = np.packbits(np.r_[False, coefficients > np.median(coefficients)])
    return int.from_bytes(bits.tobytes(), "big")


def fingerprints(image):
    gray = image.convert("L")
    views = [gray, ImageOps.mirror(gray)]
    for angle in (-15, -8, 8, 15):
        rotated = gray.rotate(angle, resample=Image.Resampling.BILINEAR)
        views.extend((rotated, ImageOps.mirror(rotated)))
    for scale in (0.80, 0.90):
        crop = ImageOps.crop(gray, tuple(round((1-scale)*d/2) for d in gray.size)*2)
        views.extend((crop, ImageOps.mirror(crop)))
    values = []
    for view in views:
        pixels = np.asarray(view.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        values.append(perceptual_hash(pixels))
    return values


def hamming(values):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    values -= (values >> 1) & np.uint64(0x5555555555555555)
    values = (values & np.uint64(0x3333333333333333)) + ((values >> 2) & np.uint64(0x3333333333333333))
    values = (values + (values >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((values * np.uint64(0x0101010101010101)) >> 56).astype(np.uint8)


def perceptual_pairs(hashes, limit):
    """Compara cada vista con la vista íntegra de las demás imágenes."""
    hashes = np.asarray(hashes, dtype=np.uint64)
    pairs = {}
    for view in hashes.T:
        for start in range(0, len(view), 128):
            distances = hamming(view[start:start+128, None] ^ hashes[None, :, 0])
            rows, cols = np.where(distances <= limit)
            for i, j in zip(rows, cols):
                a, b = sorted((start + int(i), int(j)))
                if a != b:
                    key = (a, b)
                    pairs[key] = min(pairs.get(key, 64), int(distances[i, j]))
    return pairs


def probe_recall(cache, seed, limit, image_count):
    """Prueba de sensibilidad con pares cuya relación se conoce por construcción."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(image_count, min(120, image_count), replace=False)
    hits = {}
    for index in sample:
        image = Image.fromarray(cache[index])
        original = np.asarray(fingerprints(image), dtype=np.uint64)
        width, height = image.size
        transforms = {
            "espejo": ImageOps.mirror(image),
            "brillo_70pct": ImageEnhance.Brightness(image).enhance(0.7),
            "rotacion_11_grados": image.rotate(11, resample=Image.Resampling.BILINEAR),
            "recorte_centrado_85pct": image.crop(
                (round(width*0.075), round(height*0.075), round(width*0.925), round(height*0.925))),
            "recorte_asimetrico_80pct": image.crop((0, 0, round(width*0.8), round(height*0.8))),
        }
        for name, altered in transforms.items():
            other = np.asarray(fingerprints(altered), dtype=np.uint64)
            best = min(hamming(original ^ other[0]).min(), hamming(other ^ original[0]).min())
            hits[name] = hits.get(name, 0) + int(best <= limit)
    return [(name, n, len(sample)) for name, n in hits.items()]


def components(size, edges):
    parents = list(range(size))

    def root(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, j in edges:
        a, b = sorted((root(i), root(j)))
        parents[b] = a
    return [root(i) for i in range(size)]
