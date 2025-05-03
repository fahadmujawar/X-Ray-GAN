import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_dc import Discriminator, Generator, initialize_weights
#import pytorch_fid.fid_score as fid
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms as custom_transforms


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 300
FEATURES_DISC = 64
FEATURES_GEN = 64

def load_real_images(dataset_root, image_size, batch_size):
    # Define a set of transformations to apply to the images
    transform = custom_transforms.Compose([
        custom_transforms.Resize(image_size),
        custom_transforms.ToTensor(),
        custom_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Adjust mean and std as needed
    ])

    # Create a DataLoader to load and preprocess the real images
    real_dataset = ImageFolder(root=dataset_root, transform=transform)
    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

    real_images = []
    for batch in real_dataloader:
        real_images.append(batch[0])

    real_images = torch.cat(real_images, dim=0)

    return real_images

def calculate_fid(real_images, fake_images):
    # Ensure that real_images and fake_images are on the same device (CPU or GPU)
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)

    # Calculate the FID score
    fid_score = fid.calculate_fid_given_paths(
        [real_images, fake_images], dims=2048  # Remove 'batch_size' and 'cuda'
    )
    return fid_score

def save_generated_images(images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(images):
        image = (image + 1) / 2  # Normalize the image from [-1, 1] to [0, 1]
        image = image.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
        image = (image * 255).byte()  # Convert to 8-bit image
        image = Image.fromarray(image.numpy())
        image.save(os.path.join(save_dir, f"generated_image_{i}.png"))

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# If you train on MNIST, remember to set channels_img to 1
#dataset = "D://Capstone Dataset//COVID"
#dataset = datasets.ImageFolder(root="D://Capstone Dataset//COVID", transform=transforms)
dataset = datasets.ImageFolder(root="C:\\kuval\\PES CSE\\CAPSTONE\\DATASET\\preprocess_TUBERCULOSIS", transform=transforms)

# comment mnist above and uncomment below if train on CelebA
# dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(50, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:50], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:50], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
        if epoch == NUM_EPOCHS - 1:
            with torch.no_grad():
                fake_images = gen(fixed_noise)
                # Save the generated images to a directory
                # You should also save real images for comparison
                save_generated_images(fake_images, "gen_tuberculosis")
                #real_images = "path_to_reference_dataset"

                # Calculate FID score
                #real_images = "D://Capstone Dataset//COVID"
                #fid_score = calculate_fid(real_images, "generated_images_dir")

                #print(f"Final FID Score: {fid_score}")
