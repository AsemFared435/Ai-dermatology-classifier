import os
import shutil
from sklearn.model_selection import train_test_split

# ===============================
# 1️⃣ Paths
# ===============================
raw_data_dir = r"D:\Data_Project"
classes = ['psoriasis', 'tinea circinata', 'urticaria']

processed_dir = "D:\\DEPI-Data Power\\CV _Project\\Data"
train_dir = os.path.join(processed_dir, 'train')
val_dir = os.path.join(processed_dir, 'val')
test_dir = os.path.join(processed_dir, 'test')

for split_dir in [train_dir, val_dir, test_dir]:
    for cls in classes:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

# ===============================
# 2️⃣ Collect images
# ===============================
images = []
labels = []

for cls in classes:
    cls_dir = os.path.join(raw_data_dir, cls)
    for img_name in os.listdir(cls_dir):
        images.append(os.path.join(cls_dir, img_name))
        labels.append(cls)

# ===============================
# 3️⃣ Split data (IMPORTANT PART)
# ===============================
train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    images,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

val_imgs, test_imgs, val_labels, test_labels = train_test_split(
    temp_imgs,
    temp_labels,
    test_size=0.5,
    random_state=42
)

# ===============================
# 4️⃣ Copy images
# ===============================
def copy_images(imgs, labels, target_dir):
    for img, label in zip(imgs, labels):
        shutil.copy(img, os.path.join(target_dir, label))

copy_images(train_imgs, train_labels, train_dir)
copy_images(val_imgs, val_labels, val_dir)
copy_images(test_imgs, test_labels, test_dir)

print("✅ Data preparation completed successfully!")
print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
