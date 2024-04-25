import pandas as pd
import numpy as np
from tkinter import Variable
import torch
from utils.utils import *
import sys
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_abmil import GatedAttention
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, args):
    """   
        train for a single fold
    """
    print('\nTraining!')
    writer_dir = os.path.join(args.results_dir, 'result0')
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None



    train_set, val_set, test_set = datasets
    print('Done!')

    # 为每个集创建一个DataLoader。
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    # print("Training on {} samples".format(len(train_loader)))
    # print("Validating on {} samples".format(len(val_loader)))
    # print("Testing on {} samples".format(len(test_loader)))


    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
        
    elif args.model_type == 'abmil':
        model = GatedAttention()
        model.cuda()

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    # print('\nInit Loaders...', end=' ')
    # train_loader = get_split_loader(train_set, training=True, testing = args.testing, weighted = args.weighted_sample)
    # val_loader = get_split_loader(val_set,  testing = args.testing)
    # test_loader = get_split_loader(test_set, testing = args.testing)
    # print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)
        
        elif args.model_type == 'abmil':
            train_loop_ab(epoch, model, train_loader, optimizer)
            stop = 0

        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break
    print('Success!!!')

    if args.model_type == 'abmil':
        test_loss, test_error, auc_score, precision, recall, f1 = abmil_test(model, test_loader, optimizer)

        directory = os.path.join(args.results_dir,'abmil_{}/'.format(args.runcycle))

        # 如果目录不存在，则创建它
        if not os.path.exists(directory):
            os.makedirs(directory)

        final_df = pd.DataFrame({'Loss': [test_loss], 'Test acc': [1-test_error], 
                                 'Test auc': [auc_score], 'precision': precision, 'recall': recall, 'f1': f1})
        savedStdout = sys.stdout  #保存标准输出流
        print_log = open(os.path.join(directory, 'printlog.txt'), "w")
        sys.stdout = print_log
        print(final_df)
        # 中间print的内容都被输出到printlog.txt    

        sys.stdout = savedStdout  #恢复标准输出流
        print_log.close()

        exit()


    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_checkpoint.pt")))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_checkpoint.pt"))

    _, val_error, val_auc, _, _, _, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger, precision, recall, f1 = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, precision, recall, f1

# from tqdm import tqdm
def train_loop_ab(epoch, model, train_loader, optimizer):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    model.train()

    train_loss = 0.
    train_error = 0.
    for batch_idx, (name, data, label, spatial_info) in enumerate(train_loader):
        name = '_'.join(map(str, name))  # 将元组转换为字符串

        init_image_size = [spatial_info[0][2], spatial_info[0][3]]

        w_featmap = init_image_size[0]
        h_featmap = init_image_size[1]

        feat_label = np.zeros((init_image_size[0].item() * init_image_size[1].item()))

        tau = 0.2
        eps = 1e-5
        no_binary_graph = False
        # print('the shape {}'.format(data.shape))
        # data = data.squeeze(0)

        feat_label, bipartition, data = ncut(data, [w_featmap, h_featmap], init_image_size,
                                             spatial_info, tau, eps, im_name=name,
                                             no_binary_graph=no_binary_graph, feat_label=feat_label)

        data, label = data.to(device), label.to(device)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, label)
        train_loss += loss.data[0]
        error, _ = model.calculate_classification_error(data, label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))

# def abmil_test(model, test_loader, optimizer):
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.cuda.set_device(0)  # 0是第一个GPU的索引

#     model.eval()
#     test_loss = 0.
#     test_error = 0.
#     for batch_idx, (name, data, label) in enumerate(test_loader):
#         data, label = data.to(device), label.to(device)

#         loss, attention_weights = model.calculate_objective(data, label)
#         test_loss += loss.data[0]
#         error, predicted_label = model.calculate_classification_error(data, label)
#         test_error += error

#         if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
#             bag_level = (label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))

#             print('\nTrue Bag Label, Predicted Bag Label: {}\n'.format(bag_level))

#     test_error /= len(test_loader)
#     test_loss /= len(test_loader)

#     print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))


#     return test_loss.cpu().numpy()[0], test_error
from sklearn.metrics import roc_auc_score

def abmil_test(model, test_loader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)  # 0是第一个GPU的索引

    model.eval()
    test_loss = 0.
    test_error = 0.
    all_labels = []
    all_predictions = []

    for batch_idx, (name, data, label, spatial_info) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)

        loss, attention_weights = model.calculate_objective(data, label)
        test_loss += loss.item()  # Updated to use .item()
        error, predicted_label = model.calculate_classification_error(data, label)
        test_error += error

        # Collect labels and predictions
        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predicted_label.cpu().detach().numpy())

        if batch_idx < 5:
            bag_level = (label.cpu().numpy()[0], int(predicted_label.cpu().numpy()[0]))
            print('\nTrue Bag Label, Predicted Bag Label: {}\n'.format(bag_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    # Calculate AUC
    auc_score = roc_auc_score(all_labels, all_predictions)

    precision = precision_score(all_labels, all_predictions, zero_division=1)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'
     .format(test_loss, test_error, auc_score, precision, recall, f1))

    return test_loss, test_error, auc_score, precision, recall, f1


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)  # 0是第一个GPU的索引

    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (name, data, label) in enumerate(loader):
        # print((name, data, label))
        data = data.squeeze(0)
        # print('the shape {}'.format(data.shape))
        data, label = data.to(device), label.to(device)

        print('Dimension data: {}!!!!!'.format(data.dim()))

        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)


        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_checkpoint.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (name, data, label) in enumerate(loader):
            data = data.squeeze(0)
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_checkpoint.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_predictions = np.zeros(len(loader))

    # slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (name, data, label) in enumerate(loader):
        if data.dim()>3:
            data = data.squeeze(0)
        data, label = data.to(device), label.to(device)
        # slide_id = slide_ids.iloc[batch_idx]
        slide_id = name
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        preds = Y_hat.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_predictions[batch_idx] = preds
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
        
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger, precision, recall, f1

import numpy as np
from scipy.linalg import eigh
from scipy import ndimage

def ncut(feats, dims, init_image_size, spatial_info, tau=0, eps=1e-5, im_name='', no_binary_graph=False,
         feat_label=None):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    cls_token = feats[0, 0:1, :].cpu().numpy()
    if feat_label is None:
        feat_label = []

    # padding = torch.zeros((1, init_image_size[0] * init_image_size[1], 1024), device='cuda')
    feats = feats.to('cuda')
    # print('w: {}, h: {}'.format(init_image_size[0], init_image_size[1]))
    # print('dim org: {}'.format(feats.shape))

    feats = feats[0, 0:, :]
    padding = torch.zeros((feats.size(0), 1024), device='cuda')
    padding_spa = np.zeros((feats.size(0), 4))
    cnt = 0
    for idx in range(feats.size(0)):
        flat_index = spatial_info[idx][0] * init_image_size[1] + spatial_info[idx][1]
        if feat_label[flat_index] == 1:
            # print(idx)
            continue
        padding[cnt] = feats[idx, :]
        padding_spa[cnt] = spatial_info[idx]
        cnt += 1
    feats = padding
    spatial_info = padding_spa.astype('int64')

    org_feats = feats
    feats = F.normalize(feats, p=2)

    A = torch.matmul(feats, feats.transpose(1, 0))
    # print('dim A: {}'.format(A.shape))
    # print('matrix builded.')

    A = A > tau
    A = A.float()
    # A = torch.where(A == 0, torch.tensor(eps, device=A.device), A)
    A = A + eps
    d_i = torch.sum(A, dim=1)
    D = torch.diag_embed(d_i)
    X = (D - A) / (D + eps)
    eigval, eigvec = torch.lobpcg(A=D - A, B=D, k=2, largest=False)
    second_smallest_vec = eigvec[:, 1]

    ###then sort and mark

    # num_elements = feats.size(0)
    # topp_index = int(num_elements * 0.3)
    topp_index = 10
    _, indices = torch.topk(second_smallest_vec, k=topp_index)
    remove_num = int(topp_index*0.2)
    perm = torch.randperm(topp_index)
    indices_to_keep = perm[remove_num:]
    indices = indices[indices_to_keep]
    bipartition = torch.ones_like(second_smallest_vec, dtype=torch.bool)
    bipartition[indices] = False



    # avg = torch.mean(abs(second_smallest_vec))
    # bipartition = second_smallest_vec > avg
    print('feats: {}, bipartition: {}'.format(feats.shape, bipartition.shape))

    train_set = org_feats[bipartition, :]

    # print('dim second_smallest_vec: {}'.format(second_smallest_vec.shape))
    padding_bi = torch.zeros((init_image_size[0] * init_image_size[1]), dtype=torch.bool, device='cuda')
    for idx in range(second_smallest_vec.size(0)):
        flat_index = spatial_info[idx][0] * init_image_size[1] + spatial_info[idx][1]
        padding_bi[flat_index] = bipartition[idx]
        if bipartition[idx]:
            feat_label[flat_index] = 1
    bipartition = padding_bi
    bipartition = bipartition.cpu().numpy()
    bipartition = bipartition.reshape(dims)

    return feat_label, bipartition, train_set
