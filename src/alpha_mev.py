# -*- coding: utf-8 -*-
"""alphaMEV.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iOKQZk0u7jr_jvTFFIYJUSg7wrOGGY_f
"""

import pandas
import numpy as np
import xgboost
import xgboost as xgb
import ast
import csv
import os
import logging
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def extract_data_features(tx_data):
    return np.array(
        [
            int(tx_data['from'], 0) % (2 ** 30),
            (int(tx_data['to'], 0) if tx_data['to'] is not None else 0) % (2 ** 30),
            int(tx_data['gas'], 0),
            int(tx_data['gasPrice'], 0),
            len(tx_data['input']),
            (int(tx_data['input'][:10], 0) if tx_data['input'] != '0x' else 0) % (2 ** 30),
            int(tx_data['nonce'], 0),
            1 if tx_data['to'] and tx_data['to'].lower() == '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D'.lower() else 0,
            # uniswap v2
            1 if tx_data['to'] and tx_data['to'].lower() == '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'.lower() else 0,
            # sushiswap
            1 if tx_data['to'] and tx_data['to'].lower() == '0xE592427A0AEce92De3Edee1F18E0157C05861564'.lower() else 0,
            # uniswap v3
        ])


WETH = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'.lower()
DAI = '0x6B175474E89094C44Da98b954EedeAC495271d0F'.lower()
USDC = '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'.lower()
USDT = '0xdAC17F958D2ee523a2206206994597C13D831ec7'.lower()
WBTC = '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599'.lower()


def extract_trace_features(tx_trace):
    # 0: ERROR
    # 1: CALL
    # 2: DELEGATECALL
    # 3: STATICCALL
    # 4: CREATE
    # 5: SELFDESTRUCT
    # 6: OTHERS

    # 7: ERC20 Approve 0x095ea7b3
    # 8: ERC20 Transfers 0xa9059cbb

    # 9: WETH Approve
    # 10: WETH Approve amount
    # 11: WETH Transfer
    # 12: WETH Transfer amount
    # 13: WETH Deposit 0xd0e30db0
    # 14: WETH Deposit amount
    # 15: WETH Withdraw 0x2e1a7d4d
    # 16: WETH Withdraw amount

    # 17: DAI Approve
    # 18: DAI Approve amount
    # 19: DAI Transfer
    # 20: DAI Transfer amount

    # 21: USDC Approve
    # 22: USDC Approve amount
    # 23: USDC Transfer
    # 24: USDC Transfer amount

    # 25: USDT Approve
    # 26: USDT Approve amount
    # 27: USDT Transfer
    # 28: USDT Transfer amount

    # 29: WBTC Approve
    # 30: WBTC Approve amount
    # 31: WBTC Transfer
    # 32: WBTC Transfer amount

    # 33: V1 Swap
    # 34: V2 Swap
    # 35: V3 Swap
    # 36: Curve Swap

    # 37: Value number
    # 38: Value amount

    trace_features = np.zeros(39)

    if 'error' in tx_trace:
        trace_features[0] += 1
    else:
        if 'value' in trace_features:
            trace_features[37] += 1
            trace_features[38] += float(int(tx_trace['value'], 0))
        trace_type = tx_trace['type']
        if trace_type == 'CALL':
            trace_features[1] += 1
            # Approve
            if tx_trace['input'].startswith('0x095ea7b3'):
                trace_features[7] += 1
                amount = float(int('0x' + tx_trace['input'][74:138], 0)) if len(tx_trace['input']) >= 138 else 0
                if tx_trace['to'] == WETH:
                    trace_features[9] += 1
                    trace_features[10] += amount
                elif tx_trace['to'] == DAI:
                    trace_features[17] += 1
                    trace_features[18] += amount
                elif tx_trace['to'] == USDC:
                    trace_features[21] += 1
                    trace_features[22] += amount
                elif tx_trace['to'] == USDT:
                    trace_features[25] += 1
                    trace_features[26] += amount
                elif tx_trace['to'] == WBTC:
                    trace_features[29] += 1
                    trace_features[30] += amount
            # Transfer
            elif tx_trace['input'].startswith('0xa9059cbb'):
                trace_features[8] += 1
                amount = float(int('0x' + tx_trace['input'][74:138], 0)) if len(tx_trace['input']) >= 138 else 0
                if tx_trace['to'] == WETH:
                    trace_features[11] += 1
                    trace_features[12] += amount
                elif tx_trace['to'] == DAI:
                    trace_features[19] += 1
                    trace_features[20] += amount
                elif tx_trace['to'] == USDC:
                    trace_features[23] += 1
                    trace_features[24] += amount
                elif tx_trace['to'] == USDT:
                    trace_features[27] += 1
                    trace_features[28] += amount
                elif tx_trace['to'] == WBTC:
                    trace_features[31] += 1
                    trace_features[32] += amount
            elif tx_trace['to'] == WETH:
                # Deposit
                if tx_trace['input'].startswith('0xd0e30db0'):
                    trace_features[13] += 1
                    trace_features[14] += float(int(tx_trace['value'], 0))
                # Withdraw
                elif tx_trace['input'].startswith('0x2e1a7d4d'):
                    trace_features[15] += 1
                    trace_features[16] += float(int('0x' + tx_trace['input'][10: 74], 0)) if len(
                        tx_trace['input']) >= 74 else 0
            # v1 swap
            elif tx_trace['input'].startswith('0x95e3c50b') or tx_trace['input'].startswith('0x013efd8b') or tx_trace[
                'input'].startswith('0xf39b5b9b') or tx_trace['input'].startswith('0x6b1d4db7'):
                trace_features[33] += 1
            # v2 swap
            elif tx_trace['input'].startswith('0x022c0d9f'):
                trace_features[34] += 1
            # v3 swap
            elif tx_trace['input'].startswith('0x128acb08'):
                trace_features[35] += 1
            # curve swap
            elif tx_trace['input'].startswith('0x3df02124') or tx_trace['input'].startswith('0xa6417ed6'):
                trace_features[36] += 1

        elif trace_type == 'DELEGATECALL':
            trace_features[2] += 1
        elif trace_type == 'STATICCALL':
            trace_features[3] += 1
        elif trace_type == 'CREATE':
            trace_features[4] += 1
        elif trace_type == 'SELFDESTRUCT':
            trace_features[5] += 1
        else:
            trace_features[6] += 1

    if 'calls' in tx_trace:
        for sub_trace in tx_trace['calls']:
            trace_features += extract_trace_features(sub_trace)

    return trace_features


def extract_features(dataset):
    features = []
    for tx in dataset.itertuples():
        tx_data = ast.literal_eval(tx.txData)
        tx_trace = ast.literal_eval(tx.txTrace)

        data_features = extract_data_features(tx_data)
        trace_features = extract_trace_features(tx_trace)
        reverted = np.array([1 if 'error' in tx_trace else 0])

        features.append(np.concatenate((data_features, trace_features, reverted)))
    return np.array(features)


if __name__ == "__main__":
    workspace_dir = os.pardir
    train = pandas.read_csv(os.path.join(workspace_dir, 'test-data', 'train.csv'), nrows=1000, error_bad_lines=False)
    test = pandas.read_csv(os.path.join(workspace_dir, 'test-data', 'test.csv'))
    print(train.describe())
    print(test.describe())

    print("Data read done!")

    x_train = extract_features(train)
    y_train = train['Label0']
    min_max_scaler = preprocessing.MinMaxScaler()

    x_train = pandas.DataFrame(min_max_scaler.fit_transform(x_train))

    print("Feature extraction done!")
    print(f"If any x_train is nan: {np.isnan(x_train).any()}")
    print(f"If any x_train is inf: {np.isnan(x_train).any()}")


    params = {
        'booster': 'gbtree',
        'verbosity': 2,
    }

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)
    dtrain_matrix = xgb.DMatrix(x_train, label=y_train)
    dvalid_matrix = xgb.DMatrix(x_valid, label=y_valid)
    print("D-matrix conversion done!")
    binaryModel = xgboost.train(
        params=params,
        dtrain=dtrain_matrix,
        evals=[(dvalid_matrix, "valid")],
        num_boost_round=2000,  # Maximum iterations
        early_stopping_rounds=100,
        verbose_eval=20,
    )
    print(f"{type(binaryModel)} model training finished!")

    # Apply same scalar to test features
    test_features = pandas.DataFrame(min_max_scaler.transform(extract_features(test)))
    binaryPredictions = binaryModel.predict(xgb.DMatrix(test_features)) >= 0.5
    print(pandas.DataFrame(binaryPredictions).describe())

    print(f"{type(binaryModel)} model prediction finished!")

    x_r_train = extract_features(train[train['Label0'] == True])
    y_r_train = train[train['Label0'] == True]['Label1']
    x_r_train, x_r_valid, y_r_train, y_r_valid = train_test_split(x_r_train, y_r_train, test_size=0.1)

    regressionModel = xgb.XGBRegressor(n_estimators=1000)

    regressionModel.fit(
        x_r_train, y_r_train,
        eval_set=[(x_r_valid, y_r_valid)],
        eval_metric='rmse',
        verbose=True
    )
    regressionPredictions = regressionModel.predict(test_features)

    output_file = os.path.join(workspace_dir, 'output', 'submission.csv')
    submission = csv.writer(open(output_file, 'w', encoding='UTF8'))
    for x, y in zip(binaryPredictions, regressionPredictions):
        submission.writerow([x, y])
