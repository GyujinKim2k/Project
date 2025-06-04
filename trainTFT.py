#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_tft.py

# 0) 필수 패키지 로드 및 패치
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.nn.utils import rnn

import pytorch_forecasting.utils._utils as _u
import pytorch_forecasting.utils as pf_utils
import pytorch_forecasting.models.base_model as bm

# pack_sequence / torch.cat 호환 패치

def concat_sequences_patched(sequences):
    if isinstance(sequences[0], rnn.PackedSequence):
        return rnn.pack_sequence(sequences, enforce_sorted=False)
    elif isinstance(sequences[0], torch.Tensor):
        return torch.cat(sequences, dim=0)
    elif isinstance(sequences[0], (tuple, list)):
        return tuple(
            concat_sequences_patched([seq[i] for seq in sequences])
            for i in range(len(sequences[0]))
        )
    else:
        raise ValueError("Unsupported sequence type")

# 패치 적용
_u.concat_sequences = concat_sequences_patched
pf_utils.concat_sequences = concat_sequences_patched
bm.concat_sequences = concat_sequences_patched

# 1) 기타 라이브러리 임포트
import numpy as np
import pandas as pd
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss


def main():
    # 2) 데이터 로드 및 기초 전처리
    feat_path = Path("data/feature_store/daily_features_per_all.feather")
    df = pd.read_feather(feat_path)

    cutoff = pd.Timestamp("2024-12-31")      # 포함(≤) 기준
    # cutoff = pd.Timestamp("2025-01-01")    # 미포함(<) 기준으로 자르고 싶다면 이렇게
    df = df[df["date"] <= cutoff]

    # # 2) weight: 재고 0 이면 0, 그 외 1
    # df["weight"] = np.where(df["inventory_est"] == 0, 0.0, 1.0)
    df=df[df['rev_per_vehicle']>0]

    # 1) 전체 기간(모든 date)에 걸친 일수 계산
    total_days = (df['date'].max() - df['date'].min()).days + 1

    # 2) rev_per_vehicle > 0인 행만 남기기
    df_pos = df[df['rev_per_vehicle'] > 0].copy()

    # 3) 각 그룹별로 남은(>0) 고유 날짜 수 계산
    #    transform('nunique')은 그룹별 고유 날짜 수를 원본 행 수만큼 반복 매핑
    pos_unique_dates = df_pos.groupby(['spot_id','vehicle_type'])['date'] \
                            .transform('nunique')

    # 4) 커버리지 비율 계산
    coverage = pos_unique_dates / total_days

    # 5) 비율 ≥ 0.10 (10%)인 행만 필터
    result = df_pos[coverage >= 0.40].reset_index(drop=True)

    # -- 결과 확인 --
    print(f"전체 기간 일수: {total_days}")
    print("남은 그룹 수:", 
        result[['spot_id','vehicle_type']].drop_duplicates().shape[0])
    print("최종 결과 행 수:", result.shape[0])

    df=result.copy()
    # 2) 날짜·정렬 & time_idx 생성 ---------------------------------------------
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["spot_id", "vehicle_type", "date"])

    # 하루 단위 연속 인덱스 (TFT 필수)
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days

    # 3) static / known / observed 피처 구분 -----------------------------------
    static_cats  = ["spot_id", "vehicle_type", "region_code"]

    time_known_cats = [
        "dow",        # 요일
        "week",       # ISO 주차
        "month",      # 월
        "quarter",    # 분기
        "is_weekend", # 주말 플래그
        "is_holiday"  # 공휴일 플래그
    ]
    time_known_reals = [
        "day_to_offday", # 다음 오프데이까지 남은 일수
        "offday_run",
        "rain_mm_lag1",
        "RH",        # relative humidity
        "W",          # wind speed
        "inventory_est","has_inventory",'avg_vehicle_age_inv', 'avg_standard_rate_inv',
    ]

    time_unknown_reals = [
        # lags & rolling
        "lag_1","lag_7","lag_28",
        "roll_mean_7","roll_std_7",
        # inventory
        "utilization_7",
        # coupon
        "vehicle_count","coupon_count",
        "coupon_flag_prev1","coupon_count_lag7",'coupon_per_vehicle',
        # region aggregates
        "region_total_rev","region_vehicle_count","region_rev_per_vehicle",
        # region×vehicle interaction
        "region_vehicle_mean_rev"
    ]

    target_col = "rev_per_vehicle"

    # 4) 인코더·디코더 길이 -----------------------------------------------------
    max_encoder_length = 60
    max_prediction_length = 20


    # 2) Cast those columns to string
    for col in static_cats + time_known_cats:
        df[col] = df[col].astype(str)

    # 인코더/디코더 길이
    max_encoder_length = 60
    max_prediction_length = 20

    training_cutoff = df["time_idx"].max() - max_prediction_length


    df_train = df[df.time_idx <= training_cutoff].copy()

    context_length = max_encoder_length
    prediction_length = max_prediction_length

    # 2) build the train dataset
    training = TimeSeriesDataSet(
        df_train,
        #df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        #weight="weight",
        group_ids=["spot_id","vehicle_type"],
        min_encoder_length=max_encoder_length,  
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        categorical_encoders = {
            "spot_id": NaNLabelEncoder(add_nan=True),
            "vehicle_type": NaNLabelEncoder(add_nan=True),
        },
        #min_prediction_idx     = max_encoder_len + max_prediction_len,
        static_categoricals=static_cats,
        time_varying_known_categoricals=time_known_cats,
        time_varying_known_reals=time_known_reals,
        time_varying_unknown_reals=time_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["spot_id","vehicle_type"], method="standard"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        randomize_length=False,
    )



    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )


    print(f"train series: {len(training)},  val series: {len(validation)}")

        # create dataloaders for model
    batch_size = 512  # set this between 32 to 128
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=20,
        prefetch_factor=4,        # 8×4=32 batch 미리 적재
        persistent_workers=True,
        pin_memory=True           # page-locked → memcpyAsync 속도 ↑
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=4,
        prefetch_factor=4,        # 8×4=32 batch 미리 적재
        persistent_workers=True,
        pin_memory=True           # page-locked → memcpyAsync 속도 ↑
    )

    # matmul precision
    torch.set_float32_matmul_precision('medium')

    tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.0012022644346174128,
    hidden_size=64,
    attention_head_size=6,
    dropout=0.2914981522194919,
    hidden_continuous_size=21,
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    optimizer="ranger",
    reduce_on_plateau_patience=4,
    )

    # 콜백: 체크포인트 자동 저장, 얼리스톱, LR 모니터
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='tft4-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
    )
    # configure network and trainer
    early_stop = EarlyStopping(
        monitor="val_loss",          # which metric to watch
        min_delta=1e-4,              # minimum change to qualify as an “improvement”
        patience=100,                 # how many epochs to wait after last time it improved
        verbose=True,                # print a message when stopping
        mode="min",                  # “min” because lower val_loss is better
    )
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        enable_model_summary=True,
        enable_progress_bar=True,
        gradient_clip_val=0.1561931652158813,
        #precision="transformer-engine",
        #num_sanity_val_steps=0,   
        limit_train_batches=1.0,
        callbacks=[LearningRateMonitor(), early_stop, checkpoint_callback],
        logger=TensorBoardLogger("lightning_logs"),
    )

    # # 4) 학습 재개 또는 새 학습 시작
    # last_ckpt = os.path.join('checkpoints', 'last.ckpt')
    # if os.path.exists(last_ckpt):
    #     trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=last_ckpt)
    # else:
    trainer.fit(tft, train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
