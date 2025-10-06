import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset = datasets.FashionMNIST(
        root='./data', train=training, download=True, transform=transform
    )

    loader = torch.utils.data.DataLoader(dataset, 
                                     batch_size=64,
                                     shuffle=training)
    
    return loader



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        correct = 0
        total_loss = 0
        total_samples = 0
        
        for images, labels in train_loader:
            opt.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            opt.step() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total_loss += loss.item() * len(images)
            total_samples += len(images)

        avg_loss = total_loss / total_samples
        accuracy = 100. * correct / total_samples
        print(f'Train Epoch: {epoch}   Accuracy: {correct}/{total_samples}({accuracy:.2f}%) Loss: {avg_loss:.3f}')



def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)
    
    test_loss /= total_samples
    accuracy = 100. * correct / total_samples
    
    if show_loss:
        print(f'Average loss: {test_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    model.eval()

    with torch.no_grad():
        logits = model(test_images[index].unsqueeze(0))

    prob = F.softmax(logits, dim=1).squeeze()

    top3_probs, top3_labels = torch.topk(prob, 3)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    for i in range(3):
        print(f"{class_names[top3_labels[i]]}: {top3_probs[i].item() * 100:.2f}%")



if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader(training=True)
    test_loader = get_data_loader(training=False)
    model = build_model()
    train_model(model, train_loader, criterion, T=5)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    test_images, _ = next(iter(test_loader))
    predict_label(model, test_images, index=0)
