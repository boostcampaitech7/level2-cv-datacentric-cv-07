# make a paragraph out of the word images
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from typing import Any

Document = dict[str : Image.Image, str : list[dict]]


def make_document(words, width=1080, height=1920) -> Document:
    image = Image.fromarray(
        np.random.normal(230, 6, (height, width, 3)).astype(np.uint8)
    )
    x, y = 0, 0
    for word in words:
        patch = word["patch"]
        size = word["size"]
        m = word["margin"]

        if x + size[0] > width:
            x = 0
            y += size[1]

        vs = (x + m, y + m, x + size[0] - m, y + size[1] - m)
        word["bbox"] = [vs[0], vs[1], vs[2], vs[1], vs[2], vs[3], vs[0], vs[3]]
        image.paste(patch, (x, y))
        x += size[0]

    return {"image": image, "words": words}


def draw_bbox(document: Document, ax=None) -> None:
    ax = ax or plt
    for word in document["words"]:
        bbox = word["bbox"]
        xs = [bbox[i] for i in range(0, 8, 2)] + [bbox[0]]
        ys = [bbox[i] for i in range(1, 8, 2)] + [bbox[1]]
        ax.plot(xs, ys, color="blue")
    ax.imshow(document["image"])
    ax.axis("off")


def pad_document_inplace(document: Document, pad=50, color=None) -> Document:
    """pad the document image and update the bounding boxes in-place."""
    if color is None:
        color = [64, 64, 64]
    image = cv2.copyMakeBorder(
        np.array(document["image"]),
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=[64, 64, 64],
    )
    document["image"] = Image.fromarray(image)
    for word in document["words"]:
        word["bbox"] = [v + pad for v in word["bbox"]]
    return document


def partial_copy(document: Document) -> Document:
    """
    copy some of the compoents only,
    i.e. the whole document image and bounding boxes.
    the rest are shared references.
    """
    image = document["image"].copy()
    words = [word.copy() for word in document["words"]]
    for word in words:
        word["bbox"] = word["bbox"].copy()
    return {"image": image, "words": words}


def simple_shows(args: list[dict[str:Any]], plots_per_row=4) -> None:
    len_args = len(args)
    fig_shape = (len_args // plots_per_row, plots_per_row)
    _ , axs = plt.subplots(*fig_shape, figsize=(3 * fig_shape[1], 4 * fig_shape[0]))
    for i in range(len_args):
        row, col = i // plots_per_row, i % plots_per_row
        draw_bbox(args[i]["doc"], axs[row][col])
        axs[row][col].set_title(args[i]["title"])
    plt.tight_layout()
    plt.show()


# test data for code completion assignment
test_bbox = [0, 0, 0, 1, 1, 1, 1, 0]

test_Ms = []
test_Ms.append(
    np.array(
        [
            [0.77642236, 0.69476674, 0.30843837],
            [0.38080611, 0.63221982, 0.80079036],
            [0.02284933, 0.50007051, 0.19127292],
        ]
    )
)
test_Ms.append(
    np.array(
        [
            [0.88682227, 0.05686006, 0.60093378],
            [0.02853952, 0.18661904, 0.87545493],
            [0.72744673, 0.76723366, 0.16924088],
        ]
    )
)
test_Ms.append(
    np.array(
        [
            [0.96813207, 0.76768699, 0.29149387],
            [0.07952546, 0.74971045, 0.67189814],
            [0.48540557, 0.02566273, 0.97981622],
        ]
    )
)

ref_outs = []
ref_outs.append(
    np.array(
        [
            1.6125564176344718,
            4.186637406570116,
            1.4510951613084286,
            2.072790607254025,
            2.4918027363119193,
            2.539673306238021,
            5.066548469574002,
            5.518326544386135,
        ]
    )
)
ref_outs.append(
    np.array(
        [
            3.550760194581829,
            5.172833714880234,
            0.7024150811403799,
            1.1341194283829648,
            0.9282987950505615,
            0.6554477724778408,
            1.659168737705654,
            1.0081487018650788,
        ]
    )
)
ref_outs.append(
    np.array(
        [
            0.297498520586671,
            0.685738940886286,
            1.0534092843694054,
            1.4138620912593949,
            1.3598054722905357,
            1.0068747896333512,
            0.8596827816599211,
            0.5128394921956985,
        ]
    )
)
