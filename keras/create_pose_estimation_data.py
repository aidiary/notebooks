import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

ann_file = './data/coco/annotations/person_keypoints_train2014.json'
coco = COCO(ann_file)

# personカテゴリが存在する画像のIDを抽出
img_ids = coco.getImgIds(catIds=coco.getCatIds(catNms=['person']))
img_ids = sorted(img_ids)

# 上記の画像情報を取得
img_info = coco.loadImgs(img_ids)

# CNNへの入力画像サイズ
IMG_WIDTH, IMG_HEIGHT = 368, 368
FM_WIDTH, FM_HEIGHT = 80, 80
SIGMA = 3

def load_image(img_paht):
    img = Image.open(img_path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img)
    # 1チャンネル画像は除く
    if len(img.shape) == 2:
        return None
    return img

def get_parts_annotation(info):
    persons = []

    width, height = info['width'], info['height']
    ann_ids = coco.getAnnIds(imgIds=info['id'])

    # 各人物のアノテーションリストを取得
    anns = coco.loadAnns(ann_ids)

    # 各人物のアノテーションから必要な情報を抽出
    for i in range(len(anns)):
        # キーポイントが1つもない人は無視
        if anns[i]['num_keypoints'] == 0: continue

        # 鼻のキーポイント (x,y,v) がない人は無視
        nose_keypoints = anns[i]['keypoints'][0:3]
        if nose_keypoints[2] != 2: continue

        # 0-1の相対座標に変換してから保存
        xs = nose_keypoints[0] / width
        ys = nose_keypoints[1] / height
        persons.append((xs, ys))

    return persons

def plot_persons(img, persons):
    plt.figure()

    # リサイズした画像を描画
    plt.imshow(img)
    plt.xlim((0, IMG_WIDTH))
    plt.ylim((IMG_HEIGHT, 0))
    plt.axis('off')

    # 鼻パーツを描画
    # 相対座標なのでピクセルに戻す
    for p in persons:
        xs, ys = p
        plt.plot([xs * IMG_WIDTH], [ys * IMG_HEIGHT], marker='o', markersize=3, color='red')

def plot_heatmap(img, hm):
    """
    img
    hm: (height, width, channel)
    """
    from scipy.misc import imresize
    plt.figure()
    img = imresize(img, (FM_WIDTH, FM_HEIGHT))
    plt.imshow(img)
    plt.imshow(hm[:, :, 0], alpha=0.4)

def create_heatmap(persons, width, height, sigma):
    """
    persons: relative keypoints list
    width: heatmap width
    height: heatmap height
    sigma
    """
    # 今回はパーツは1つなのでSは3次元でOK
    S = np.zeros(shape=(len(persons), height, width))

    for k in range(len(persons)):  # 各人について処理
        x = np.array(persons[k])
        assert len(x) == 2

        # relative => pixel
        x[0] = width * x[0]
        x[1] = height * x[1]

        for i in range(height):
            for j in range(width):
                p = np.array([j, i])
                d = np.linalg.norm(p - x, 2) ** 2
                S[k, i, j] = np.exp(- d / (sigma * sigma))

    # 人物単位でmaxをとる
    hm = S.max(axis=0)

    # チャネルを追加
    # 鼻のみなので1チャンネル
    hm = np.expand_dims(hm, axis=2)

    return hm


# 画像をデータ化して保存
input_img_data = []  # (batch, height, width, channel)
heatmaps = []        # (batch, height, width, channel=1)
filenames = []

for info in img_info[:10]:  # 各画像についてループ
    # パーツ情報（鼻の相対座標）を取得
    persons = get_parts_annotation(info)
    # パーツが1つもない画像は無視
    if persons == []: continue

    # 画像をロードして配列可して保存
    img_path = os.path.join('data/coco/train2014', info['file_name'])
    img = load_image(img_path)
    if img is None: continue
    input_img_data.append(img)
    filenames.append(info['file_name'])

    # テスト用のプロット
#     plot_persons(img, persons)

    # ヒートマップを作成
    # パーツの座標はCNNの出力するヒートマップサイズに合わせた座標に変換する
    hm = create_heatmap(persons, FM_WIDTH, FM_HEIGHT, SIGMA)
    heatmaps.append(hm)

    # 元画像を縮小した画像に重ね合わせる
    plot_heatmap(img, hm)

input_img_data = np.array(input_img_data)
heatmaps = np.array(heatmaps)
filenames = np.array(filenames)

print(input_img_data.shape)
print(heatmaps.shape)
print(filenames)

np.save('input_img_data.npy', input_img_data)
np.save('heatmaps.npy', heatmaps)
np.save('filenames.npy', filenames)
