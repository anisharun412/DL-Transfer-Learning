# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset

### Problem Statement

The objective of this experiment is to develop an image classification model using Transfer Learning with the VGG19 architecture. Instead of training a deep neural network from scratch, a pre-trained VGG19 model is used as a feature extractor and its final classification layer is modified to classify images from the CHIP dataset. The model is trained and evaluated using training and testing image datasets, and its performance is measured using accuracy, confusion matrix, and classification report.

### Dataset

The experiment uses the CHIP Dataset (chip_data.zip), which contains images organized into separate folders corresponding to different classes.

## Neural Network Model

### VGG19 Transfer Learning Architecture
<img width="850" height="540" alt="vgg-architecture" src="https://github.com/user-attachments/assets/1de3d38c-a3a6-454d-8665-0615db07af78" />

## DESIGN STEPS

STEP 1:
Import the required PyTorch, Torchvision, NumPy, Matplotlib, and Scikit-Learn libraries.

STEP 2:
Load and preprocess the image dataset using ImageFolder and apply image transformations.

STEP 3:
Create DataLoaders for training and testing datasets.

STEP 4:
Load the pre-trained VGG19 model and replace the final classification layer according to the dataset classes.

STEP 5:
Freeze the convolutional feature extraction layers and define the loss function and optimizer.

STEP 6:
Train the transfer learning model using the training dataset and monitor training and validation loss.

STEP 7:
Evaluate the model using the testing dataset.

STEP 8:
Generate confusion matrix and classification report.

STEP 9:
Predict the class of unseen images using the trained model.

STEP 10:
Visualize the prediction results and analyze model performance.

## PROGRAM

### Name: Arunsamy D

### Register Number: 212224240016

```python
# Load Pretrained Model and Modify for Transfer Learning

model = models.vgg19(pretrained=True)

# Modify the final fully connected layer to match the dataset classes

num_features = model.classifier[6].in_features

model.classifier[6] = nn.Linear(
    num_features,
    len(train_dataset.classes)
)

# Include the Loss function and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier[6].parameters(),
    lr=0.001
)

# Train the model

## Step 3: Train the Model

def train_model(model, train_loader,test_loader,num_epochs=10):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        model.train()

        running_train_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion( outputs, labels )

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()

        train_losses.append( running_train_loss / len(train_loader) )

        # Compute validation loss
        
        model.eval()

        running_val_loss = 0.0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion( outputs, labels )

                running_val_loss += loss.item()

        val_losses.append( running_val_loss / len(test_loader) )

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print('Name: Arunsamy D')
    print('Register Number: 212224240016')
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT

### Sample Images from Dataset
<img width="407" height="109" alt="download" src="https://github.com/user-attachments/assets/74db6b3f-dc01-46ff-9aea-2ce9296b1ab5" />

### Training Dataset Information
<img width="918" height="67" alt="image" src="https://github.com/user-attachments/assets/0a8c7e8f-d8dd-4450-a377-922c25fb3e87" />

### Testing Dataset Information
<img width="1406" height="73" alt="image" src="https://github.com/user-attachments/assets/02e13aee-83e9-4635-96c0-7a9067e21289" />

### VGG19 Model Summary
<img width="993" height="1189" alt="image" src="https://github.com/user-attachments/assets/5c8dfed0-9a83-44ae-9453-e05d8f1b7354" />

### VGG19 Model Summary After modifing the final fully connected layer to match the dataset classes
<img width="1015" height="1199" alt="image" src="https://github.com/user-attachments/assets/5aaf675b-6f4a-4a4c-8789-adedeaaf7ec8" />

### Model Trainging & Training Loss vs Validation Loss Plot

<img width="780" height="251" alt="image" src="https://github.com/user-attachments/assets/f88ff043-bb28-4050-844c-805e34b1a49f" />
<img width="691" height="547" alt="download" src="https://github.com/user-attachments/assets/48af2b98-914e-4be6-8a77-f45f8c9cb950" />

### Test Accuracy

<img width="735" height="68" alt="image" src="https://github.com/user-attachments/assets/808d781c-217b-4ff8-9ee9-d3f6eaa5a538" />

### Confusion Matrix
<img width="640" height="547" alt="download" src="https://github.com/user-attachments/assets/98a0d8fd-b84a-4099-9d65-0a72bba9128a" />

### Classification Report
<img width="771" height="254" alt="image" src="https://github.com/user-attachments/assets/080c630e-804c-4d13-9142-f7de23a3d4ad" />

### Sample Prediction 1
<img width="748" height="549" alt="image" src="https://github.com/user-attachments/assets/34dfbb85-41d1-4652-8e6e-65e257cae438" />

### Sample Prediction 2
<img width="760" height="528" alt="image" src="https://github.com/user-attachments/assets/aebd159f-9c42-422f-b992-c8945ce78859" />

## RESULT

Thus, an image classification model was successfully developed using Transfer Learning with the VGG19 architecture. The model was trained on the CHIP dataset, evaluated using accuracy, confusion matrix, and classification report, and successfully classified images with high accuracy.
