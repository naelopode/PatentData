#import fitz  # PyMuPDF
from PIL import Image
from torch.utils.data import Dataset
import io
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
#from pdf2image import convert_from_path
from pypdf import PdfReader
import tarfile
import subprocess
import os
from PIL import Image
from torch.utils.data import DataLoader
from pymongo import MongoClient

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((64, 64)),  # Resize all images to 32x32
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

def extract_images_from_pdf(pdf_path, output_prefix="image", output_dir="/scratch/students/ndillenb/tmp2"):
    # Make sure output dir exists
    new_path = os.path.join(output_dir, pdf_path.split("/")[-1].split(".pdf")[0])
    os.makedirs(new_path, exist_ok=True)

    # Output file base path (no extension)
    output_base = os.path.join(new_path, output_prefix)
    # Run pdfimages (ppm output is default for most formats)
    subprocess.run(["pdfimages", "-tiff", pdf_path, output_base], check=True)

def read_extracted(pdf_path, page_number, output_dir):
    new_path = os.path.join(output_dir, pdf_path.split("/")[-1].split(".pdf")[0])
    extracted_files = sorted([
        os.path.join(new_path, f)
        for f in os.listdir(new_path)
        if f.startswith("image")
    ], key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    # Load all images using PIL
    file = extracted_files[page_number]
    try:
        img = Image.open(file)
        image =img.copy()  # Copy so we can safely delete files later
        img.close()
        return image
    except Exception as e:
        print(f"Could not open {file}: {e}")

def delete_folder(folder):
    if os.path.exists(folder):
        for root, dirs, files in os.walk(folder, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(folder)


def extract_tif_from_tar_gz(tar_gz_path, output_dir="/scratch/students/ndillenb/tmp2"):
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tifs = []
        for member in tar.getmembers():
            if member.name.endswith(".tif"):
                tifs.append(os.path.join(tar_gz_path + "/" + member.name))
        return sorted(tifs, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))) # Sorted paths to extracted PDF
    return []  # No tiff found

class PDFPageDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths  # list of PDF file paths
        self.transform = transform
        self.pages = []
        self.errors = 0
        self.errors_list = []
        # Index all pages (pdf_path, page_number)
        for pdf_path in tqdm(self.file_paths):
            if pdf_path.endswith('.tar.gz'):
                pdf_path = extract_tif_from_tar_gz(pdf_path)
                for i, item in enumerate(pdf_path):
                    self.pages.append((item, i, 'tif', False))
            elif pdf_path.endswith('.pdf'):
                try:
                    reader = PdfReader(pdf_path)
                    for page_number in range(len(reader.pages)):
                        last = True if page_number == len(pdf_path) - 1 else False
                        self.pages.append((pdf_path, page_number, 'pdf', last))
                except Exception as e:
                    print(f"Error reading {pdf_path}: {e}")

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, idx):
        file_path, page_number, type, last = self.pages[idx]
        if type == 'pdf':
            # Load the specific page from PDF
            try:
                if page_number == 0:
                    extract_images_from_pdf(file_path)
                image = read_extracted(file_path, page_number, "/scratch/students/ndillenb/tmp2")
                #images = convert_from_path(str(file_path), dpi='50', first_page=page_number+1, last_page=page_number+1)
                #image = images[0]
                if last:
                    delete_folder("/scratch/students/ndillenb/tmp2/" + file_path.split("/")[-1].split(".pdf")[0])
            except Exception as e:
                image = Image.new('RGB', (64, 64), (255, 255, 255))  # Create an empty white image
                self.errors += 1
                self.errors_list.append(self.pages[idx])
        elif type == 'tif':
            tar_path = file_path.split("/")[:-1]
            tar_gz_path = "/".join(tar_path)
            tif_name = file_path.split("/")[-1]
            try:
                with tarfile.open(tar_gz_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name==tif_name:
                            f=tar.extractfile(member)
                            content=f.read()
                            image = Image.open(io.BytesIO(content))
            except Exception as e:
                image = Image.new('RGB', (64, 64), (255, 255, 255))
                self.errors += 1
                self.errors_list.append(self.pages[idx])
        if self.transform:
            image = self.transform(image)
        return image

pdfs_path = '/lhstdata1/patentdata/raw/gb/scans'
# Process all PDFs in subfolders and store images in a dictionary
pdf_files = []
for pdf_file in glob.glob(os.path.join(pdfs_path, '**/*.pdf'), recursive=True):
    pdf_files.append(pdf_file)
tar_files = []
for tar_file in glob.glob(os.path.join(pdfs_path, '**/*.tar.gz'), recursive=True):
    tar_files.append(tar_file)
files = tar_files + pdf_files
files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))) # Sorted paths to extracted PDF
print('Loading DataLoader...')
apply_loader = torch.load('/scratch/students/ndillenb/clustering/CNN/gb_patents_loader.pth', weights_only = False)
BATCH_SIZE = 64
print('Loading DataLoader done')

# Define CNN Model


class_names = ['partial_schema', 'schema', 'text']

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjusted for 32x32 input size
        self.fc2 = nn.Linear(512, len(class_names))  # Output layer with 5 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model
model = CNN().to(device)

# Load the model
model_path = "/scratch/students/ndillenb/clustering/CNN/model_grayscale_3class_64x64_2.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded")
# MongoDB connection

client = MongoClient("localhost", 29012)
db = client["test-database"]
collection_CNN = db["CNN_GB"]

print("MongoDB connection established")

with torch.no_grad():
    updates = []
    for i, item in tqdm(enumerate(apply_loader)):
        paths = apply_loader.dataset.pages[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
        images = item.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for j in range(len(predicted)):
            white_pixels = torch.sum(images[j] == 1).item()
            if white_pixels > (images[j].numel() * 0.96):  
                if ".tar.gz" in paths[j][0]:
                    publication_number = paths[j][0].split('/')[-2].replace('.tar.gz','')
                else:
                    publication_number = paths[j][0].split('/')[-1].replace('.pdf','')
                updates.append({"path": paths[j][0], "Publication Number" : publication_number, "page":paths[j][1], "type": 'blank'})
            else:
                if ".tar.gz" in paths[j][0]:
                    publication_number = paths[j][0].split('/')[-2].replace('.tar.gz','')
                else:
                    publication_number = paths[j][0].split('/')[-1].replace('.pdf','')
                updates.append({"path": paths[j][0], "Publication Number" : publication_number, "page":paths[j][1], "type": class_names[predicted[j]]})
        if len(updates) > 30:
            collection_CNN.insert_many(updates)
            updates = []
    if len(updates) > 0:
        collection_CNN.insert_many(updates)
print("Data inserted into MongoDB")