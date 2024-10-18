import torch.optim
import numpy as np
import os
from tqdm.auto import tqdm
# from tqdm import tqdm  <<- 연구는 py 포멧이 아닌 주피터로 수행하였으므로 notebook용 tqdm 사용을 위해 주석처리
from copy import copy
from packages.sampler import DataSampling
from packages.dlmodels import WPDataset, WODataset, CNN, LSTM, ResNet18
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error # MAPE 추가 호출
from sklearn.preprocessing import StandardScaler
# 더 나은 훈련과정 시각화를 위하여
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


########################################################################################################################
# integrating codes
# Modifier : Jongwook Kim
# Last update : 01-Jan-2023
########################################################################################################################
# Required initial args's instance
# args.silent : ProxyModel의 훈련 과정에서 tqdm을 사용할지 안할지에 대한 boolean 형태의 instance
########################################################################################################################
class ProxyModel:
    def __init__(self,
                 args,
                 positions,
                 setting,
                 ):
        self.setting = setting
        self.model_name = setting['Model']

        if setting['Model'] == 'CNN':
            self.model = CNN(input_flag=setting['Input'])
            # self.model = CNN() # 입력 차원 조절을 위하여 args argument가 추가됨
        elif setting['Model'] == 'ResNet':
            self.model = ResNet18(input_flag=setting['Input'])
            # self.model = ResNet18() # 입력 차원 조절을 위하여 args argument가 추가됨
        elif setting['Model'] == 'LSTM':
            self.model = LSTM(
                            input_size=positions[0].num_of_wells,
                            hidden_size=100,
                            output_size=3,
                            sequence_length=len(positions[0].wells[0].control),
                            num_layers=2)

########################################################################################################################
# initialization changes
# add
#  + self.input : Saving the input features of ProxyModel's objects to their instance "input"
#  + self.device : Switching the torch device to assigned device for training Deep learning models
#  + self.metric : add the key "MAPE; mean absolute percentage error" : value "empty list"
########################################################################################################################
        self.args = args
        self.saved_dir = f"{args.train_model_saved_dir}/{setting['Model']}"
        self.input = self.setting['Input']
        self.device = self.setting['Device']
        self.data_sampling = DataSampling(args)
        self.perm = self.data_sampling.perm
        self.MaxTOF = args.max_tof
        self.MaxP = args.max_pressure
        self.positions = positions
        self.fit_mean = np.mean(np.array([d.fit for d in positions]))
        self.fit_std = np.std(np.array([d.fit for d in positions]))
        for position in positions:
            position.fit_norm = (position.fit - self.fit_mean) / self.fit_std
        self.metric = {"r2_score": [], "MAPE": []}

    def preprocess(self, data, model_name):
        def __merge_schedule_control__(schedule, control):
            return [s * c for s, c in zip(schedule, control)]

        args = self.args

        scaler_input = StandardScaler()
        scaler_output = StandardScaler()
        if model_name == 'LSTM':
            production_time = args.production_time
            production_steps = int(production_time / args.tstep)
            drilling_steps = int(production_time / args.dstep)

            # normalize input data
            data_input = []
            for d in data:
                scheduled_control = [__merge_schedule_control__(well.schedule, well.control) for well in d.wells]
                data_input.append(np.array(scheduled_control).reshape(-1, ))
            data_input = np.array(data_input)
            scaler_input.fit(data_input)
            data_input_norm = scaler_input.transform(data_input)

            input_preprocessed = []
            for d, norm in zip(data, data_input_norm):
                norm = norm.reshape(len(d.wells), -1)
                input_preprocessed.append(norm)
                for well, data_input in zip(d.wells, norm):
                    well.control_norm = data_input.tolist()

            # normalize output data
            data_output = []
            for d in data:
                prod_all = np.array(d.prod_data.filter(regex='T_discounted').dropna()).transpose()
                time_index = [idx for idx in range(0, production_steps, int(production_steps / drilling_steps))]
                prod_all = prod_all[:, time_index].reshape(-1, )
                data_output.append(prod_all)
            data_output = np.array(data_output)
            scaler_output.fit(data_output)
            data_output_norm = scaler_output.transform(data_output)

            output_preprocessed = []
            for d, norm in zip(data, data_output_norm):
                norm = norm.reshape(3, -1)
                d.prod_data_norm = norm.tolist()
                output_preprocessed.append(norm)

        elif model_name in ['CNN', 'ResNet']:
            data_output = [d.fit for d in data]
            data_output = np.array(data_output).reshape(-1, 1)
            scaler_output.fit(data_output)
            data_output_norm = scaler_output.transform(data_output)
            for d, norm in zip(data, data_output_norm):
                d.fit_norm = norm[0]

        self.scaler = scaler_output

        return data

########################################################################################################################
# make_dataloader changes

# test_ratio에서 발생한 오류 수정을 위해 기존 코드 제외
# WPDataset 부분 수정 (argument의 수가 추가되었으므로)
# random_split에서 컴퓨터의 수 표현에서 발생한 오류를 방지하기 위해 int -> round로 수정

########################################################################################################################
# 01/15 재수정
# 소수점 자리가 0.5보다 이하이면 또 에러가 발생하므로, train_ratio와 validation ratio를 이용하여 train, validation dataset의 개수를
# 지정해준 뒤, 총 데이터셋에서 그 개수를 빼어 test dataset의 개수를 지정해 줌
########################################################################################################################
    def make_dataloader(self, data, train_ratio, validate_ratio):
        args = self.args

        data = self.preprocess(data, model_name=self.model_name)
        if self.model_name in ['CNN', 'ResNet']:
            dataset = WPDataset(data=data, maxtof=args.max_tof, maxP=args.max_pressure, res_oilsat=args.res_oilsat,
                                nx=args.num_of_x, ny=args.num_of_y, transform=None,
                                flag_input=self.setting['Input'])
        elif self.model_name in ['LSTM']:
            dataset = WODataset(data, args.production_time, args.dstep, args.tstep, None)
        else:
            NotImplementedError('Model not supported')

        # 1/15 sequence 추가
        # 22222222222222222222222222222222222222222222222222222222222222
        if not self.setting['Valid_pick']:
            train_ratio += validate_ratio
            validate_ratio = 0
        # 22222222222222222222222222222222222222222222222222222222222222
        sequence = [int(round(len(dataset) * ratio)) for ratio in [train_ratio, validate_ratio]]
        sequence.append(int(len(dataset) - sum(sequence)))
        train_dataset, validation_dataset, test_dataset = random_split(dataset, sequence)
        train_dataloader = DataLoader(train_dataset, batch_size=self.setting['Batch_size'], shuffle=True)
        # 22222222222222222222222222222222222222222222222222222222222222
        if not self.setting['Valid_pick']:
            valid_dataloader = []
        else:
            valid_dataloader = DataLoader(validation_dataset, batch_size=self.setting['Batch_size'], shuffle=True)
        # 22222222222222222222222222222222222222222222222222222222222222
        test_dataloader = DataLoader(test_dataset, batch_size=self.setting['Batch_size'], shuffle=True)

        return train_dataloader, valid_dataloader, test_dataloader

    def train_model(self, data, train_ratio=0.7, validate_ratio=0.15, saved_dir='./model',
                    saved_model='saved_model'):

        train_dataloader, valid_dataloader, test_dataloader = \
            self.make_dataloader(data, train_ratio=train_ratio, validate_ratio=validate_ratio)

        if self.setting['Line search']:
            list_bs = [20, 40, 60, 80, 100, 150, 200]
            list_lr = [1e-3, 3e-3]
            self.dict_model = {'batch_size': [], 'lr': [], 'model': []}
            metric = 0
            model_now = copy(self.model)
            for lr in list_lr:
                for bs in list_bs:
                    print(f'Batch size: {bs} | Lr: {lr} \n')
                    model_tmp = self.train(model_now, train_dataloader, valid_dataloader, test_dataloader,
                                                saved_dir, saved_model)
                    self.dict_model['batch_size'].append(bs)
                    self.dict_model['lr'].append(lr)
                    self.dict_model['model'].append(model_tmp)
                    if self.metric['r2_score'][-1] > metric:
                        self.model = model_tmp
                    if self.setting['Silent']:
                        print(f"R2 Score:{self.metric['r2_score'][-1]:.4f} | MAPE: {self.metric['MAPE'][-1]:.1f}")
        else:
            self.model = self.train(self.model, train_dataloader, valid_dataloader, test_dataloader, saved_dir,
                                    saved_model)
        return self.model

    ########################################################################################################################
    # train changes
    # 훈련할 모델과 자료, 그리고 loss function에 device 할당
    # agrs에 silent instance가 지정되어 있으면 tqdm으로, 아니면 tqdm off {args.silent의 dtype은 boolean}
    ########################################################################################################################
    def train(self, model, train_dataloader, valid_dataloader, test_dataloader, saved_dir='./model',
              saved_model='saved_model'):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.setting['Lr'])
        if self.setting['Scheduler'] == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.setting['Gamma'])
        elif self.setting['Scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20,
                                                        gamma=self.setting['Gamma'])
        elif self.setting['Scheduler'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10,
                                                                   factor=self.setting['Gamma'])
        elif self.setting['Scheduler'] == 'LambdaLR':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lr_lambda=lambda epoch: self.setting['Gamma'] ** epoch)

        if self.setting['Tensorboard']['Use']:
            writer = SummaryWriter(f"./logs/{self.model_name}/{self.setting['Tensorboard']['Filename']}")
        min_valid_loss = np.inf
        model.to(self.device)
        criterion.to(self.device)

        # shuffle train / validation dataloader for each epoch

        eps = 1e-7
        if not self.setting['Silent']:
            iter_bar = tqdm(range(self.setting['Epoch']))
        else:
            iter_bar = range(self.setting['Epoch'])
        for epoch in iter_bar:
            model.train()
            train_loss = 0.0
            # shuffle train / validation dataloader for each epoch
            for batch in train_dataloader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()

                pred = model(x)

                loss = torch.sqrt(criterion(pred.squeeze(), y) + eps)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            if self.setting['Scheduler']:
                if self.setting['Scheduler'] == 'ReduceLROnPlateau':
                    pass
                else:
                    scheduler.step()
            if not self.setting['Silent']: iter_bar.set_description(
                f"epoch: {epoch} - loss: {train_loss / len(train_dataloader):.4f} ")
            # 11111111111111111111111111111111111111111111111111111111111111111
            if not self.setting['Valid_pick']:
                train_loss /= len(train_dataloader)
                if self.setting['Tensorboard']['Use']:
                    writer.add_scalars(self.setting['Tensorboard']['Tagname'],
                                       {'train': train_loss}, epoch + 1)
                if not self.setting['Silent']:
                    print(
                        f'Epoch {epoch + 1} \t\t Training Loss: {train_loss:.4f}')
            elif self.setting['Valid_pick']:
                # 11111111111111111111111111111111111111111111111111111111111111111
                valid_loss = 0.0
                model.eval()  # Optional when not using Model Specific layer
                model.to(self.device)
                for batch in valid_dataloader:
                    x_v, y_v = batch
                    x_v, y_v = x_v.to(self.device), y_v.to(self.device)

                    target = model(x_v)
                    loss = torch.sqrt(criterion(target.squeeze(), y_v) + eps)

                    valid_loss += loss.item()

                train_loss /= len(train_dataloader)
                valid_loss /= len(valid_dataloader)
                if self.setting['Scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                if self.setting['Tensorboard']['Use']:
                    writer.add_scalars(self.setting['Tensorboard']['Tagname'],
                                       {'train': train_loss,
                                        'validation': valid_loss}, epoch + 1)
                if not self.setting['Silent']:
                    print(
                        f'Epoch {epoch + 1} \t\t Training Loss: {train_loss:.4f} \t\t '
                        f'Validation Loss: {valid_loss:.4f}')

                if min_valid_loss > valid_loss:
                    if self.setting['Valid_pick']:
                        if not self.setting['Silent']: print(
                            f'Validation Loss Decreased({min_valid_loss:.4f}--->{valid_loss:.4f}) \t Saving The Model')
                        min_valid_loss = valid_loss
                        # Saving State Dict
                        if not os.path.exists(saved_dir):
                            os.mkdir(saved_dir)
                        torch.save(model.state_dict(), f'{saved_dir}/{saved_model}.pth')
            # 11111111111111111111111111111111111111111111111111111111111111111
        if not self.setting['Silent']: print(f'Now test to test_dataset')
        if self.setting['Valid_pick']:
            model.load_state_dict(torch.load(f'{saved_dir}/{saved_model}.pth'))
        else:
            torch.save(model.state_dict(), f'{saved_dir}/{saved_model}.pth')
        if self.setting['Tensorboard']['Use']:
            writer.close()
        self.inference(model, test_dataloader)

        return model

########################################################################################################################
# inference changes
# 검증할 모델과 자료에 device 할당
# GPU 이용 시 데이터의 자료형이 다를 수 있으므로, target과 y_t에 detach()와 cpu(), 그리고 numpy()를 붙혀줌
# metric instance에 추가적으로 MAPE를 저장, 그리고 커맨드 창에 도시하도록 함
########################################################################################################################
    def inference(self, model, dataloader, label_exist=True):
        model.eval()
        model.to(self.device)

        predictions = []
        reals = []
        for batch in dataloader:
            x_t, y_t = batch
            x_t, y_t = x_t.to(self.device), y_t.to(self.device)

            target = model(x_t)
            if target.dim() > 2:
                target = target.view(target.shape[0], -1)
                y_t = y_t.view(y_t.shape[0], -1)

            prediction = self.scaler.inverse_transform(target.detach().cpu().numpy())
            if y_t.dim() == 1:
                y_t = y_t.reshape(-1, 1)
            real = self.scaler.inverse_transform(y_t.detach().cpu().numpy())
            predictions.extend(prediction)
            reals.extend(real)

        if label_exist:
            self.predictions = predictions
            self.reals = reals
            self.metric['r2_score'].append(r2_score([r for r in reals], [p for p in predictions]))
            self.metric['MAPE'].append(100 * mean_absolute_percentage_error(reals, predictions))
            if not self.setting['Silent']:
                print(f"R2 Score:{self.metric['r2_score'][-1]:.4f}")
                print(f"MAPE: {self.metric['MAPE'][-1]:.1f}%")

        self.test_loss = np.sqrt(mean_squared_error(reals, predictions))
        return predictions, reals