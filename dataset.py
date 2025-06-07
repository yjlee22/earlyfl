import os
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
import medmnist
from PIL import Image
from medmnist.info import INFO
import glob
import random

class DatasetObject:
    def __init__(self, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path='', args=None):
        self.args = args
        self.n_client = n_client
        self.rule = rule
        self.rule_arg = rule_arg
        self.seed = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        dataset_name_lower = "chestmnist"
        self.name = f"{dataset_name_lower}_{self.n_client}_{self.seed}_{self.rule}_{rule_arg_str}"
        self.name += f"_{unbalanced_sgm:f}" if unbalanced_sgm != 0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.channels = 1
        self.width = 28
        self.height = 28
        self.set_data()
       
    def set_data(self):
        save_path = os.path.join(self.data_path, 'Data', self.name)
        client_x_path = os.path.join(save_path, 'client_x.npy')
        client_y_path = os.path.join(save_path, 'client_y.npy')
        test_x_path = os.path.join(save_path, 'test_x.npy')
        test_y_path = os.path.join(save_path, 'test_y.npy')

        if not (os.path.exists(client_x_path) and os.path.exists(client_y_path) and 
                os.path.exists(test_x_path) and os.path.exists(test_y_path)):
            os.makedirs(save_path, exist_ok=True)
            
            DataClass = medmnist.ChestMNIST
            info = INFO[DataClass.flag]

            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])

            train_dataset = DataClass(split='train', transform=transform, download=True, as_rgb=False)
            test_dataset = DataClass(split='test', transform=transform, download=True, as_rgb=False)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=len(train_dataset), shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=len(test_dataset), shuffle=False)
            
            self.channels = 1
            self.width = 28
            self.height = 28
            self.n_cls = len(info['label'])

            train_data = next(iter(train_loader))
            test_data = next(iter(test_loader))
            
            train_x, train_y = train_data[0].numpy(), train_data[1].numpy()
            test_x, test_y = test_data[0].numpy(), test_data[1].numpy()
            
            print("Train X shape:", train_x.shape)
            print("Train Y shape:", train_y.shape)
            print("Test X shape:", test_x.shape)
            print("Test Y shape:", test_y.shape)

            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(train_y))
            train_x = train_x[rand_perm]
            train_y = train_y[rand_perm]

            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y

            # Calculate data distribution
            n_data_per_client = int(len(train_y) / self.n_client)
            client_data_list = np.ones(self.n_client, dtype=int) * n_data_per_client
            diff = np.sum(client_data_list) - len(train_y)
            
            # Adjust data distribution to match dataset size
            if diff != 0:
                for client_i in range(self.n_client):
                    if client_data_list[client_i] > diff:
                        client_data_list[client_i] -= diff
                        break

            # Data distribution based on the specified rule
            if self.rule in ['Dirichlet', 'Pathological']:
                pseudo_classes = np.zeros(len(train_y), dtype=int)
                
                if self.rule == 'Dirichlet':
                    
                    label_counts = np.sum(train_y, axis=0)
                    top_labels = np.argsort(label_counts)[::-1][:5] 
                    
                    for i, y in enumerate(train_y):
                        cls = 0
                        for bit, label in enumerate(top_labels):
                            if y[label] == 1:
                                cls += (1 << bit) 
                        pseudo_classes[i] = cls
                    
                    n_pseudo_cls = (1 << len(top_labels))
                    cls_priors = np.random.dirichlet(alpha=[self.rule_arg] * n_pseudo_cls, size=self.n_client)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                
                elif self.rule == 'Pathological':
                    
                    label_counts = np.sum(train_y, axis=0)
                    sorted_labels = np.argsort(label_counts)[::-1]
                    
                    c = int(self.rule_arg)
                    label_groups = np.array_split(sorted_labels, c)
                    
                    for i, y in enumerate(train_y):
                        scores = [np.sum(y[group]) for group in label_groups]
                        pseudo_classes[i] = np.argmax(scores) if max(scores) > 0 else 0
                    
                    n_pseudo_cls = c
                    a = np.ones([self.n_client, n_pseudo_cls])
                    a[:, c::] = 0
                    [np.random.shuffle(i) for i in a]
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i, j] != 0:
                                prior_cumsum[i, j] = a[i, 0:j+1].sum() / c * 1.0
                
                idx_list = [np.where(pseudo_classes == i)[0] for i in range(max(pseudo_classes) + 1)]
                cls_amount = [len(idx_list[i]) for i in range(len(idx_list))]
                
                true_sample = [0 for i in range(len(idx_list))]
                
                client_x = [np.zeros((client_data_list[client__], 1, self.height, self.width), dtype=np.float32)
                            for client__ in range(self.n_client)]
                client_y = [np.zeros((client_data_list[client__], self.n_cls), dtype=np.float32)
                            for client__ in range(self.n_client)]

                while np.sum(client_data_list) != 0:
                    curr_client = np.random.randint(self.n_client)
                    if client_data_list[curr_client] <= 0:
                        continue
                    client_data_list[curr_client] -= 1
                    curr_prior = prior_cumsum[curr_client]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        if cls_label >= len(idx_list) or cls_amount[cls_label] <= 0:
                            # Skip if class label is out of range or no more samples
                            if cls_label < len(idx_list):
                                cls_amount[cls_label] = len(idx_list[cls_label])
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1
                        client_x[curr_client][client_data_list[curr_client]] = train_x[idx_list[cls_label][cls_amount[cls_label]]]
                        client_y[curr_client][client_data_list[curr_client]] = train_y[idx_list[cls_label][cls_amount[cls_label]]]
                        break

                print("Sample distribution per class:", true_sample)
                self.client_x = client_x
                self.client_y = client_y

            elif self.rule == 'iid':
                # For IID, simply distribute data evenly
                client_x = [np.zeros((client_data_list[client__], 1, self.height, self.width), dtype=np.float32)
                            for client__ in range(self.n_client)]
                client_y = [np.zeros((client_data_list[client__], self.n_cls), dtype=np.float32)
                            for client__ in range(self.n_client)]
                client_data_list_cum_sum = np.concatenate(([0], np.cumsum(client_data_list)))
                for client_idx_ in range(self.n_client):
                    client_x[client_idx_] = train_x[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                    client_y[client_idx_] = train_y[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                self.client_x = client_x
                self.client_y = client_y

            self.test_x = test_x
            self.test_y = test_y
            
            # Save data to files
            np.save(client_x_path, np.array(self.client_x, dtype=object))
            np.save(client_y_path, np.array(self.client_y, dtype=object))
            np.save(test_x_path, test_x)
            np.save(test_y_path, test_y)
            
            # Print shapes to verify
            print("Client X shapes:", [x.shape for x in self.client_x[:2]])
            print("Client Y shapes:", [y.shape for y in self.client_y[:2]])
            print("Test X shape:", self.test_x.shape)
            print("Test Y shape:", self.test_y.shape)
        else:
            # Load data from existing files
            self.client_x = np.load(client_x_path, allow_pickle=True)
            self.client_y = np.load(client_y_path, allow_pickle=True)
            self.n_client = len(self.client_x)
            self.test_x = np.load(test_x_path, mmap_mode='r')
            self.test_y = np.load(test_y_path, mmap_mode='r')
            
            # Get dataset properties
            ds_class_name = "ChestMNIST"
            DataClass = getattr(medmnist, ds_class_name)
            info = INFO[DataClass.flag]
            self.channels = 1
            self.width = 28
            self.height = 28
            self.n_cls = len(info['label'])
            
            # Print shapes to verify
            print("Client X shapes:", [x.shape for x in self.client_x[:2]])
            print("Client Y shapes:", [y.shape for y in self.client_y[:2]])
            print("Test X shape:", self.test_x.shape)
            print("Test Y shape:", self.test_y.shape)
        
        # Process synthetic data if needed for early stopping or pretraining
        if hasattr(self.args, 'syn') and self.args.syn:
            # Load or create synthetic data
            self.load_synthetic_data()
    
    def load_synthetic_data(self):
        syn_data_dir = os.path.join(self.data_path, 'Data')
        os.makedirs(syn_data_dir, exist_ok=True)
        syn_x_path = os.path.join(syn_data_dir, f"syn_{self.args.generator}_{self.args.num_per_class}_x.npy")
        syn_y_path = os.path.join(syn_data_dir, f"syn_{self.args.generator}_{self.args.num_per_class}_y.npy")
        
        if os.path.exists(syn_x_path) and os.path.exists(syn_y_path):
            print(f"Loading pre-processed synthetic data for {self.args.generator} with {self.args.num_per_class} images per class")
            self.syn_x = np.load(syn_x_path, allow_pickle=True)
            self.syn_y = np.load(syn_y_path, allow_pickle=True)
            
            print(f"Synthetic X shape: {self.syn_x.shape}")
            print(f"Synthetic Y shape: {self.syn_y.shape}")
        else:
            print(f"Creating synthetic data for {self.args.generator} with {self.args.num_per_class} images per class")
            self._create_synthetic_dataset(syn_x_path, syn_y_path)
    
    def _create_synthetic_dataset(self, syn_x_path, syn_y_path):
        syn_transform = transforms.Compose([
            transforms.Resize((28, 28)), 
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        DataClass = medmnist.ChestMNIST
        info = INFO[DataClass.flag]
        self.n_cls = len(info['label'])  
        
        chest_class_folders = [
            "atelectasis", "cardiomegaly", "effusion", "infiltration",
            "mass", "nodule", "pneumonia", "pneumothorax", "consolidation",
            "edema", "emphysema", "fibrosis", "pleural", "hernia"
        ]
                
        ordered_class_names = []
        for i in range(len(info['label'])):
            ordered_class_names.append(info['label'][str(i)])
        
        folder_to_index = {}
        for i, class_name in enumerate(ordered_class_names):
            folder_to_index[class_name.lower()] = i
        
        all_syn_images = []
        all_syn_labels = []
        
        # Base path for synthetic data
        syn_base_path = os.path.join(os.path.dirname(self.data_path), 'synthetic_data', self.args.generator)
        print(f"Looking for synthetic images in: {syn_base_path}")
        
        # For each class folder, load the synthetic images
        for class_folder in chest_class_folders:
            # Path to synthetic data for this class
            syn_class_path = os.path.join(syn_base_path, class_folder)
            
            if not os.path.exists(syn_class_path):
                print(f"Warning: Path {syn_class_path} does not exist")
                continue
            
            # Get class index for this folder
            if class_folder in folder_to_index:
                class_idx = folder_to_index[class_folder]
            else:
                print(f"Warning: Class folder {class_folder} not in mapping. Available mappings:")
                for folder, idx in folder_to_index.items():
                    print(f"  {folder} -> {idx}")
                continue
            
            # Get image paths for this class - randomly selecting images
            image_paths = []
            all_image_files = glob.glob(os.path.join(syn_class_path, '*.png'))
            
            if all_image_files:
                # Randomly select images if there are enough
                if len(all_image_files) > self.args.num_per_class:
                    image_paths = random.sample(all_image_files, self.args.num_per_class)
                else:
                    # Use all available images if there aren't enough
                    image_paths = all_image_files
                
                print(f"Randomly selected {len(image_paths)} out of {len(all_image_files)} images for class {class_folder}")
            else:
                print(f"Warning: No images found for class {class_folder} in {syn_class_path}")
                continue
            
            # Process each image
            successful_images = 0
            for img_path in image_paths:
                try:
                    # Load image
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    
                    # Apply synthetic transform directly to PIL image
                    img_tensor = syn_transform(img)
                    
                    # Convert tensor to numpy
                    img_np = img_tensor.numpy()
                    
                    # Add to list
                    all_syn_images.append(img_np)
                    
                    # Create one-hot label (multi-label classification)
                    label = np.zeros(self.n_cls, dtype=np.float32)
                    label[class_idx] = 1.0
                    all_syn_labels.append(label)
                    
                    successful_images += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            print(f"Successfully added {successful_images} images for class {class_folder} (index {class_idx})")
        
        # Check if we have any images
        if not all_syn_images:
            raise ValueError("No synthetic images were loaded. Check paths and file format.")
        
        # Convert lists to numpy arrays - store as simple arrays, not client/test split
        self.syn_x = np.array(all_syn_images)
        self.syn_y = np.array(all_syn_labels)
        
        print(f"Loaded total of {len(all_syn_images)} synthetic images")
        print(f"Synthetic X shape: {self.syn_x.shape}")
        print(f"Synthetic Y shape: {self.syn_y.shape}")
        
        # Save the synthetic data
        np.save(syn_x_path, self.syn_x)
        np.save(syn_y_path, self.syn_y)
        
        print(f"Saved synthetic data as {syn_x_path} and {syn_y_path}")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y=True, train=False, dataset_name='', args=None):
        self.name = dataset_name
        self.args = args
        self.train = train
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        self.X_data = data_x
        self.y_data = data_y if isinstance(data_y, bool) else data_y.astype('float32')
        self.channels = 1
        self.width = 28
        self.height = 28
           
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        img = self.X_data[idx].copy()
        
        if getattr(self.args, 'syn', False):
            img = np.clip(img, 0.0, 1.0)
        
        img = np.moveaxis(img, 0, -1)
        img_pil = Image.fromarray((img * 255).astype(np.uint8).squeeze())
        img = self.transform(img_pil)
        
        if img.shape[0] != 1:
            print(f"Warning: Image has {img.shape[0]} channels after transform, expected 1")
            
        return img if isinstance(self.y_data, bool) else (img, self.y_data[idx])


class DatasetFromDir(data.Dataset):
    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]
        img_raw = Image.open(img_path).convert('L')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list)