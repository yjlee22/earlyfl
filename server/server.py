import os
import time
import numpy as np

import torch
from torch.utils import data

from utils import *
from dataset import Dataset

class Server(object):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        self.args = args
        self.device = device
        self.datasets = datasets
        self.model_func = model_func
        
        self.server_model = init_model
        self.server_model_params_list = init_par_list

        # Each client has a local model param vector
        self.clients_params_list = init_par_list.repeat(args.total_client, 1)
        self.clients_updated_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        self.test_perf = np.zeros((self.args.comm_rounds, 2))  # [loss, acc]
        
        self.time = np.zeros((args.comm_rounds))
        self.lr = self.args.local_learning_rate

        self.comm_vecs = {'Params_list': None}
        self.received_vecs = None
        self.Client = None
        
        # Early stopping related variables - changed to track accuracy instead of loss
        self.best_val_acc = 0.0  # Changed to track accuracy instead of loss
        self.patience_counter = 0
        self.early_stopped = False
        self.best_model_params = init_par_list.clone()
        self.early_stopping_round = 0

    def _activate_clients_(self, t):
        # randomly pick active clients
        return np.random.choice(
            range(self.args.total_client),
            max(int(self.args.active_ratio * self.args.total_client), 1),
            replace=False
        )

    def _lr_scheduler_(self):
        self.lr *= self.args.lr_decay

    def compute_metrics(self, preds, targets):
        """
        Compute only loss and accuracy for multi-label classification.
        
        Args:
            preds: Tensor with shape (batch_size, num_classes) of predicted probabilities
            targets: Tensor with shape (batch_size, num_classes) of binary targets (0 or 1)
            
        Returns:
            Dictionary with accuracy
        """
        # Apply threshold to convert probabilities to binary predictions
        threshold = 0.5
        binary_preds = (preds >= threshold).float()
        
        # Sample-based accuracy (correctly predicted elements / total elements)
        accuracy = torch.sum(binary_preds == targets).float() / (targets.shape[0] * targets.shape[1])
        accuracy = accuracy.item()
        
        return {
            'accuracy': accuracy
        }

    def _validate_(self, dataset):
        """
        Evaluate the global server model on the test set using:
          - BCEWithLogitsLoss for test_loss
          - Accuracy
        """
        self.server_model.eval()
        
        testloader = data.DataLoader(
            Dataset(dataset[0], dataset[1], train=False,
                    dataset_name=self.args.dataset, args=self.args),
            batch_size=256, shuffle=False
        )
        
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        
        total_loss = 0.0
        batch_count = 0

        # For accumulating predictions and targets
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(self.device)          
                labels = labels.to(self.device)            
                logits = self.server_model(inputs)        
                loss = loss_fn(logits, labels)       
                total_loss += loss.item()
                batch_count += 1

                # Probability predictions via sigmoid
                probs = torch.sigmoid(logits)
                
                # Store for metric calculation
                all_probs.append(probs.cpu())
                all_targets.append(labels.cpu())

        # Concatenate all batches
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = self.compute_metrics(all_probs, all_targets)
        test_loss = total_loss / batch_count
        
        return (
            test_loss, 
            metrics['accuracy']
        )
    
    def _validate_synthetic_(self):
        """
        Evaluate the global server model on synthetic validation data for early stopping only
        """
        if not hasattr(self.datasets, 'syn_x') or not hasattr(self.datasets, 'syn_y'):
            print("Warning: Synthetic data not available for early stopping")
            return float('inf'), 0.0  # Return high loss for missing data
            
        self.server_model.eval()
        
        # Create a validation loader with synthetic data
        valloader = data.DataLoader(
            Dataset(self.datasets.syn_x, self.datasets.syn_y, train=False,
                    dataset_name='synthetic', args=self.args),
            batch_size=256, shuffle=False
        )
        
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        
        total_loss = 0.0
        batch_count = 0

        # For accumulating predictions and targets
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(self.device)          
                labels = labels.to(self.device)            
                logits = self.server_model(inputs)        
                loss = loss_fn(logits, labels)       
                total_loss += loss.item()
                batch_count += 1

                # Probability predictions via sigmoid
                probs = torch.sigmoid(logits)
                
                # Store for metric calculation
                all_probs.append(probs.cpu())
                all_targets.append(labels.cpu())

        # Concatenate all batches
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = self.compute_metrics(all_probs, all_targets)
        val_loss = total_loss / batch_count
        
        return (
            val_loss, 
            metrics['accuracy']
        )
        
    def _early_stopping_check_(self, t):
        """
        Check if training should be stopped early based on validation accuracy
        on synthetic data
        """
        if not self.args.early:
            return False
            
        # Evaluate on synthetic validation data
        val_loss, val_acc = self._validate_synthetic_()
        
        print(f"    Synthetic Val (Early Stopping) --- Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
        
        # Check if validation accuracy improved (higher is better)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.patience_counter = 0
            self.best_model_params = self.server_model_params_list.clone()
            self.early_stopping_round = t
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.args.patience:
                print(f"\n=== Early stopping triggered at round {t+1} ===")
                print(f"=== Best validation accuracy on synthetic data: {self.best_val_acc*100:.2f}% at round {self.early_stopping_round+1} ===")
                
                # Restore best model parameters
                self.server_model_params_list = self.best_model_params.clone()
                set_client_from_params(self.device, self.server_model, self.server_model_params_list)
                
                self.early_stopped = True
                return True
                
        return False

    def process_for_communication(self, client, Averaged_update):
        pass
        
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedAvg approach
        new_params_list = self.server_model_params_list + Averaged_update
        return new_params_list
    
    def postprocess(self, client, received_vecs):
        pass

    def _test_(self, t, selected_clients):
        loss, acc = self._validate_(
            (self.datasets.test_x, self.datasets.test_y)
        )
        self.test_perf[t] = [loss, acc]
        
        print("    Test    ----    Loss: {:.4f},   Acc: {:.2f}%".format(
            loss, acc*100
        ))

    def _save_results_(self):
        
        if not self.args.non_iid:
            root = f'{self.args.out_file}/IID'
        else:
            root = f'{self.args.out_file}/{self.args.split_rule}_{self.args.split_coef}'
            
        if not os.path.exists(root):
            os.makedirs(root)
            
        # Add early stopping, pretraining, imagenet info to filename
        syn_suffix = f"_syn_{self.args.num_per_class}" if self.args.syn else ""
        early_suffix = f"_early_{self.args.patience}" if self.args.early else ""
        out_path = root + f'/{self.args.method}_generator_{self.args.generator}{syn_suffix}{early_suffix}.npy'
        np.save(out_path, self.test_perf)

    def _summary_(self):
        if not self.args.non_iid:
            summary_root = f'{self.args.out_file}/summary/IID'
        else:
            summary_root = f'{self.args.out_file}/summary/{self.args.split_rule}_{self.args.split_coef}'
            
        if not os.path.exists(summary_root):
            os.makedirs(summary_root)
            
        # Add early stopping, pretraining, imagenet info to filename
        syn_suffix = f"_syn_{self.args.num_per_class}" if self.args.syn else ""
        early_suffix = f"_early_{self.args.patience}" if self.args.early else ""
        
        summary_file = summary_root + f'/{self.args.method}_generator_{self.args.generator}{syn_suffix}{early_suffix}.txt'

        with open(summary_file, 'w') as f:
            f.write("##=============================================##\n")
            f.write("##                   Summary                   ##\n")
            f.write("##=============================================##\n")
            f.write("Communication round   --->   T = {:d}\n".format(self.args.comm_rounds))
            f.write("Average Time / round   --->   {:.2f}s \n".format(np.mean(self.time[:self.early_stopping_round+1 if self.early_stopped else self.args.comm_rounds])))
            
            if self.early_stopped:
                f.write("Early stopped at round   --->   {:d}\n".format(self.early_stopping_round + 1))
                f.write("Best validation accuracy on synthetic data   --->   {:.2f}%\n".format(self.best_val_acc * 100))
                
            f.write("Top-1 Test Acc (T)    --->   {:.2f}% ({:d})".format(np.max(self.test_perf[:,1]) * 100, np.argmax(self.test_perf[:,1])))
        
        print("##=============================================##")
        print("##                   Summary                   ##")
        print("##=============================================##")
        print(f"Communication rounds   --->   T = {self.early_stopping_round+1 if self.early_stopped else self.args.comm_rounds}")
        print(f"Average Time / round   --->   {np.mean(self.time[:self.early_stopping_round+1 if self.early_stopped else self.args.comm_rounds]):.2f}s")

        if self.early_stopped:
            print(f"Early stopped at round   --->   {self.early_stopping_round + 1}")
            print(f"Best validation accuracy on synthetic data   --->   {self.best_val_acc*100:.2f}%")

        # self.test_perf columns: [loss, acc]
        best_acc = np.max(self.test_perf[:,1])
        best_acc_round = np.argmax(self.test_perf[:,1])

        print(f" Top-1 Accuracy        --->  {best_acc*100:.2f}% (round {best_acc_round+1})")

        # Print final round performance
        final_round = self.early_stopping_round if self.early_stopped else self.args.comm_rounds - 1
        print("\nFinal Round Performance:")
        print(f" Accuracy        --->  {self.test_perf[final_round,1]*100:.2f}%")

    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")

        # Print early stopping setting
        if self.args.early:
            print(f"Early stopping enabled with patience {self.args.patience} based on accuracy using synthetic data")

        Averaged_update = torch.zeros(self.server_model_params_list.shape, device=self.device)
        
        for t in range(self.args.comm_rounds):
            start = time.time()
            selected_clients = self._activate_clients_(t)
            print(f'============= Communication Round {t+1} =============')
            print('Selected Clients:', selected_clients)

            for client_id in selected_clients:
                dataset_tuple = (self.datasets.client_x[client_id], self.datasets.client_y[client_id])
                self.process_for_communication(client_id, Averaged_update)

                # Instantiate & train that client's model
                _edge_device = self.Client(
                    device=self.device,
                    model_func=self.model_func,
                    received_vecs=self.comm_vecs,
                    dataset=dataset_tuple,
                    lr=self.lr,
                    args=self.args
                )
                self.received_vecs = _edge_device.train()
                self.clients_updated_params_list[client_id] = self.received_vecs['local_update_list']
                self.clients_params_list[client_id] = self.received_vecs['local_model_param_list']
                self.postprocess(client_id, self.received_vecs)
                del _edge_device

            # Average updates
            subset_updates = self.clients_updated_params_list[selected_clients]
            Averaged_update = torch.mean(subset_updates, dim=0)

            subset_models = self.clients_params_list[selected_clients]
            Averaged_model = torch.mean(subset_models, dim=0)

            # Update global model
            self.server_model_params_list = self.global_update(selected_clients, Averaged_update, Averaged_model)
            set_client_from_params(self.device, self.server_model, self.server_model_params_list)

            # Evaluate on test set
            self._test_(t, selected_clients)
            
            # Check for early stopping using synthetic data
            if self._early_stopping_check_(t):
                # Fill remaining test_perf with final values for plotting
                if t < self.args.comm_rounds - 1:
                    self.test_perf[t+1:] = self.test_perf[t]
                break

            # Adjust LR
            self._lr_scheduler_()
            
            end = time.time()
            self.time[t] = end - start
            print(f"            ----    Time: {self.time[t]:.2f}s")

        self._save_results_()
        self._summary_()