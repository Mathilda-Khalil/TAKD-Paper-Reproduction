import os
import nni
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import get_cifar
from model_factory import create_cnn_model, is_resnet

SAVE_DIR = "/content/drive/MyDrive/TAKD_models"
os.makedirs(SAVE_DIR, exist_ok=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def parse_arguments():
    parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: cifar10 or cifar100')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay')
    parser.add_argument('--teacher', default='', type=str, help='teacher model name')
    parser.add_argument('--student', '--model', default='resnet8', type=str, help='student model name')
    parser.add_argument('--teacher-checkpoint', default='', type=str, help='optional pretrained checkpoint for teacher')
    parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not to use cuda')
    parser.add_argument('--dataset-dir', default='./data', type=str, help='dataset directory')
    args = parser.parse_args()
    return args


def load_checkpoint(model, checkpoint_path, device='cpu'):
    model_ckp = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_ckp['model_state_dict'])
    return model


class TrainManager(object):
    def __init__(self, student, teacher=None, train_loader=None, test_loader=None, train_config=None):
        if train_config is None:
            train_config = {}

        self.student = student
        self.teacher = teacher
        self.have_teacher = self.teacher is not None
        self.device = train_config['device']
        self.name = train_config['name']

        self.optimizer = optim.SGD(
            self.student.parameters(),
            lr=train_config['learning_rate'],
            momentum=train_config['momentum'],
            weight_decay=train_config['weight_decay'],
            nesterov=True
        )

        if self.have_teacher:
            self.teacher.eval()
            self.teacher.train(mode=False)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = train_config

    def train(self):
        lambda_ = self.config['lambda_student']
        T = self.config['T_student']
        epochs = self.config['epochs']
        trial_id = self.config['trial_id']

        best_acc = 0.0
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.student.train()
            self.adjust_learning_rate(self.optimizer, epoch)

            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output = self.student(data)

                loss_sl = criterion(output, target)
                loss = loss_sl

                if self.have_teacher:
                    with torch.no_grad():
                        teacher_outputs = self.teacher(data)

                    loss_kd = nn.KLDivLoss(reduction='batchmean')(
                        F.log_softmax(output / T, dim=1),
                        F.softmax(teacher_outputs / T, dim=1)
                    )
                    loss = (1 - lambda_) * loss_sl + lambda_ * (T * T) * loss_kd

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(self.train_loader):.4f}")
            val_acc = self.validate(step=epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                self.save(epoch, name=f'{self.name}_{trial_id}_best.pth.tar')

        return best_acc

    def validate(self, step=0):
        self.student.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.student(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = 100.0 * correct / total
            print(f'{{"metric": "{self.name}_val_accuracy", "value": {acc}}}')
            return acc

    def save(self, epoch, name=None):
        trial_id = self.config['trial_id']
        if name is None:
            save_path = os.path.join(SAVE_DIR, f'{self.name}_{trial_id}_epoch{epoch}.pth.tar')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)
        else:
            save_path = os.path.join(SAVE_DIR, name)
            torch.save({
                'model_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
            }, save_path)

    def adjust_learning_rate(self, optimizer, epoch):
      epochs = self.config['epochs']
      model_is_plain = self.config['is_plane']

      if model_is_plain:
        # FIX for CNN (plane models)
        if epoch < int(epochs * 0.5):
            lr = 0.01
        elif epoch < int(epochs * 0.75):
            lr = 0.001
        else:
            lr = 0.0001
      else:
        base_lr = self.config['learning_rate']
        if epoch < int(epochs * 0.6):
            lr = base_lr
        elif epoch < int(epochs * 0.8):
            lr = base_lr * 0.1
        else:
            lr = base_lr * 0.01

      for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    config = nni.get_next_parameter()
    if not config:
        config = {
            'seed': 42,
            'T_student': 4,
            'lambda_student': 0.9,
        }

    trial_id = os.environ.get('NNI_TRIAL_JOB_ID', 'manual')

    use_cuda = args.cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    torch.manual_seed(config['seed'])
    if use_cuda:
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])

    dataset = args.dataset.lower()
    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'cifar10':
        num_classes = 10
    else:
        raise ValueError("Dataset must be either 'cifar10' or 'cifar100'")

    teacher_model = None
    student_model = create_cnn_model(args.student, dataset, use_cuda=use_cuda)

    train_config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'device': device,
        'is_plane': not is_resnet(args.student),
        'trial_id': trial_id,
        'T_student': config.get('T_student', 4),
        'lambda_student': config.get('lambda_student', 0.9),
    }

    train_loader, test_loader = get_cifar(num_classes)

    if args.teacher:
        teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=use_cuda)

        if args.teacher_checkpoint:
            print("---------- Loading Teacher -------")
            teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint, device=device)
        else:
            print("---------- Training Teacher -------")
            teacher_train_config = copy.deepcopy(train_config)
            teacher_train_config['name'] = args.teacher

            teacher_trainer = TrainManager(
                teacher_model,
                teacher=None,
                train_loader=train_loader,
                test_loader=test_loader,
                train_config=teacher_train_config
            )
            teacher_trainer.train()

            teacher_name = os.path.join(SAVE_DIR, f'{args.teacher}_{trial_id}_best.pth.tar')
            teacher_model = load_checkpoint(teacher_model, teacher_name, device=device)

    print("---------- Training Student -------")
    student_train_config = copy.deepcopy(train_config)
    student_train_config['name'] = args.student

    student_trainer = TrainManager(
        student_model,
        teacher=teacher_model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_config=student_train_config
    )

    best_student_acc = student_trainer.train()

    try:
        nni.report_final_result(best_student_acc)
    except Exception:
        print(f"Final result: {best_student_acc:.2f}")