from imports import*
from models import*

# Training and evaluation loop
num_epochs = 200
patience = 50

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        all_train_labels = []
        all_train_preds = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.double())
            loss = criterion(outputs.double(), labels.unsqueeze(1).double())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct_train += (preds == labels.unsqueeze(1)).sum().item()
            total_train += labels.size(0)

            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(outputs.detach().cpu().numpy())  # Use detach() here
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.double())
                loss = criterion(outputs.double(), labels.unsqueeze(1).double())
                running_val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct_val += (preds == labels.unsqueeze(1)).sum().item()
                total_val += labels.size(0)

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(outputs.detach().cpu().numpy())  # Use detach() here
        
        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_wts = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_model_wts)
    return model
    
# Function to evaluate the model
def evaluate_model(model, dataloader, criterion):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.double())
            loss = criterion(outputs.double(), labels.unsqueeze(1).double())
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.detach().cpu().numpy())
    
    accuracy = correct / total

    return running_loss / len(dataloader), accuracy, all_labels, all_preds

# Perform grid search for hyperparameter tuning
def grid_search(train_loader, val_loader, num_epochs, patience):
    
    conv_kernel_size = [5,10,15]

    hidden_dim_CNN_LSTM = [28,56,112]

    fc_hidden_dims_CNN_LSTM = [[28, 16], [56, 32], [112, 64]]

    fc_hidden_dims_MLP = [[140, 120, 90, 45], [120, 100, 80, 40], [100, 50, 25, 10]]

    width = [100,80,50]

    grid = [2,3,5]

    order = [2,3,5]

    best_val_auc = 0
    best_hyperparams = None
    best_model_wts = None

    for width, grid, order in itertools.product(width, grid, order):
        
        print(f"Training with fc_hidden_dims={width}, grid={grid}, order={order}")

        # Initialize model with current hyperparameters
        model = KAN(width, grid, order)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience)

        # Evaluate on validation set
        val_loss, val_accuracy, y_val_true, y_val_pred = evaluate_model(model, val_loader, criterion)
        val_auc = roc_auc_score(y_val_true, y_val_pred)

        # Keep track of the best performing model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_hyperparams = {
                'width': width,
                'grid': grid,
                'order': order 
            }
            best_model_wts = model.state_dict()

    print(f"Best Hyperparameters: {best_hyperparams}")
    print(f"Best Validation AUC: {best_val_auc:.4f}")

    # Load best model weights
    model = KAN(best_hyperparams['width'], best_hyperparams['grid'], best_hyperparams['order'])
    model.load_state_dict(best_model_wts)
    return model, best_hyperparams