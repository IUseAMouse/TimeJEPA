"""
Test complet: loader → transformation → forward → backward
Adapté pour JEPATST
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# The masked-JEPA API (`model(x, context_mask=..., target_mask=...)`) and the
# `create_jepa_tst_*` factories were removed in favour of Hydra configs and a
# pure forecasting-JEPA objective. Tests below that still depend on them are
# marked deprecated rather than deleted.
DEPRECATED_MASKED_API = pytest.mark.skip(
    reason="Legacy masked-JEPA API (context_mask/target_mask, create_jepa_tst_*) "
           "no longer exists; see tests/test_p0_regressions.py"
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.dataset import TimeSeriesDataset
from timejepa.data.datamodule import MonashDataModule
from timejepa.data.normalizer import get_normalizer
from timejepa.models.jepa_tst import JEPATST


def test_normalizer_identity():
    """Vérifie que identity normalizer ne modifie pas les données."""
    print("\n" + "="*80)
    print("TEST 1: Identity Normalizer")
    print("="*80)
    
    data = np.random.randn(10, 100).astype(np.float32)
    
    # Test identity
    normalizer = get_normalizer("identity")
    normalizer.fit(data)
    transformed = normalizer.transform(data)
    
    assert np.allclose(data, transformed), "Identity should not change data!"
    print("✅ Identity normalizer: data unchanged")
    
    normalizer_std = get_normalizer("standard")
    normalizer_std.fit(data)
    transformed_std = normalizer_std.transform(data)
    
    assert not np.allclose(data, transformed_std), "Standard should change data!"
    assert np.abs(transformed_std.mean()) < 1e-5, "Mean should be ~0"
    assert np.abs(transformed_std.std() - 1.0) < 0.1, "Std should be ~1"
    print("✅ Standard normalizer: data normalized correctly")
    print(f"   Original: mean={data.mean():.3f}, std={data.std():.3f}")
    print(f"   Standard: mean={transformed_std.mean():.6f}, std={transformed_std.std():.3f}")


def test_dataset_with_normalizers():
    """Test dataset avec différents normalizers."""
    print("\n" + "="*80)
    print("TEST 2: Dataset avec normalizers")
    print("="*80)
    
   
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        data = np.random.randn(5, 200).astype(np.float32)
        np.save(f.name, data)
        temp_path = Path(f.name)
    
    try:
        
        dataset_identity = TimeSeriesDataset(
            data_path=temp_path,
            context_length=50,
            prediction_length=10,
            stride=10,
            normalizer=None,  
            normalize_mode="global",
        )
        
        sample = dataset_identity[0]
        context = sample['context'].numpy()
        
       
        print(f"✅ Identity dataset:")
        print(f"   Context shape: {context.shape}")
        print(f"   Context mean: {context.mean():.3f}, std: {context.std():.3f}")
        print(f"   Original mean: {data.mean():.3f}, std: {data.std():.3f}")
        print(f"   Normalizer: {dataset_identity.normalizer.__class__.__name__}")
        

        normalizer_std = get_normalizer("standard")
        dataset_standard = TimeSeriesDataset(
            data_path=temp_path,
            context_length=50,
            prediction_length=10,
            stride=10,
            normalizer=normalizer_std,
            normalize_mode="global",
        )
        
        sample_std = dataset_standard[0]
        context_std = sample_std['context'].numpy()
        
        print(f"\n✅ Standard dataset:")
        print(f"   Context mean: {context_std.mean():.6f}, std: {context_std.std():.3f}")
        print(f"   Normalizer: {dataset_standard.normalizer.__class__.__name__}")
        
        assert not np.allclose(context, context_std), "Identity vs Standard should differ!"
        
    finally:
        temp_path.unlink()


def test_datamodule_normalizer_types():
    """Test MonashDataModule avec différents normalizer_type."""
    print("\n" + "="*80)
    print("TEST 3: MonashDataModule normalizer_type")
    print("="*80)
    
    # Créer données temporaires
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        data = np.random.randn(5, 200).astype(np.float32)
        np.save(f.name, data)
        temp_path = Path(f.name)
    
    try:
        # Test identity
        dm_identity = MonashDataModule(
            data_path=temp_path,
            context_length=50,
            prediction_length=10,
            batch_size=2,
            normalizer_type="identity",
            normalize_mode="global",
            num_workers=0,
        )
        dm_identity.prepare_data()
        dm_identity.setup()
        
        print(f"✅ DataModule with identity:")
        print(f"   Normalizer: {dm_identity.normalizer.__class__.__name__}")
        assert dm_identity.normalizer.__class__.__name__ == "IdentityNormalizer"
        
        # Test standard
        dm_standard = MonashDataModule(
            data_path=temp_path,
            context_length=50,
            prediction_length=10,
            batch_size=2,
            normalizer_type="standard",
            normalize_mode="global",
            num_workers=0,
        )
        dm_standard.prepare_data()
        dm_standard.setup()
        
        print(f"\n✅ DataModule with standard:")
        print(f"   Normalizer: {dm_standard.normalizer.__class__.__name__}")
        assert dm_standard.normalizer.__class__.__name__ == "StandardScaler"
        
        # Comparer les données
        batch_identity = next(iter(dm_identity.train_dataloader()))
        batch_standard = next(iter(dm_standard.train_dataloader()))
        
        print(f"\n✅ Batch comparison:")
        print(f"   Identity batch mean: {batch_identity['context'].mean():.3f}")
        print(f"   Standard batch mean: {batch_standard['context'].mean():.6f}")
        
    finally:
        temp_path.unlink()


@DEPRECATED_MASKED_API
def test_full_pipeline_with_model():
    """Test complet: dataloader → model → forward → backward."""
    print("\n" + "="*80)
    print("TEST 4: Pipeline complète avec JEPATST Tiny")
    print("="*80)
    
    # Créer données temporaires
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        data = np.random.randn(10, 500).astype(np.float32)
        np.save(f.name, data)
        temp_path = Path(f.name)
    
    try:
        # Setup datamodule
        dm = MonashDataModule(
            data_path=temp_path,
            context_length=128,
            prediction_length=32,
            batch_size=4,
            stride=16,
            normalizer_type="identity",  # 🔥 Test avec identity
            normalize_mode="global",
            num_workers=0,
            persistent_workers=False,
        )
        dm.prepare_data()
        dm.setup()
        
        print(f"✅ DataModule setup:")
        print(f"   Train samples: {len(dm.train_dataset)}")
        print(f"   Normalizer: {dm.normalizer.__class__.__name__}")
        
        # Get a batch
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"\n✅ Batch loaded:")
        print(f"   Context shape: {batch['context'].shape}")
        print(f"   Target shape: {batch['target'].shape}")
        print(f"   Context mean: {batch['context'].mean():.3f}, std: {batch['context'].std():.3f}")
        
        # Create model (tiny version)
        model = create_jepa_tst_tiny()
        model.set_pretrain_mode(True)  # Mode pretrain
        
        print(f"\n✅ Model created (JEPATST Tiny):")
        param_counts = model.get_num_params()
        for name, count in param_counts.items():
            print(f"   {name}: {count:,}")
        
        # Forward pass
        context = batch['context']  # (B, T)
        if context.dim() == 2:
            context = context.unsqueeze(-1)  # (B, T, C=1)
        
        print(f"\n✅ Forward pass:")
        print(f"   Input shape: {context.shape}")
        
        # Create mask pour JEPA
        B, T, C = context.shape
        num_patches = (T - model.patch_size) // model.stride + 1
        
        # Context mask: premiers 60% des patches
        n_context = int(num_patches * 0.6)
        context_mask = torch.zeros(B, num_patches, dtype=torch.bool)
        context_mask[:, :n_context] = True
        
        # Target mask: derniers 40%
        target_mask = torch.zeros(B, num_patches, dtype=torch.bool)
        target_mask[:, n_context:] = True
        
        print(f"   Num patches: {num_patches}")
        print(f"   Context patches: {n_context}, Target patches: {num_patches - n_context}")
        
        # Forward
        output = model(
            x=context,
            context_mask=context_mask,
            target_mask=target_mask,
        )
        
        print(f"   Output predictions shape: {output['predictions'].shape}")
        print(f"   Output targets shape: {output['targets'].shape}")
        print(f"   Output context_embeddings shape: {output['context_embeddings'].shape}")
        
        # Compute loss
        predictions = output['predictions']
        targets = output['targets']
        
        loss = torch.nn.functional.mse_loss(predictions, targets)
        print(f"   Loss: {loss.item():.6f}")
        
        # Backward pass
        print(f"\n✅ Backward pass:")
        loss.backward()
        
        # Check gradients
        has_grads = sum(1 for p in model.parameters() if p.grad is not None)
        total_params_count = sum(1 for _ in model.parameters())
        print(f"   Parameters with gradients: {has_grads}/{total_params_count}")
        
        # Vérifier que online encoder a des gradients mais pas target encoder
        online_encoder_grads = sum(1 for p in model.online_encoder.parameters() if p.grad is not None)
        target_encoder_grads = sum(1 for p in model.target_encoder.parameters() if p.grad is not None)
        
        print(f"   Online encoder grads: {online_encoder_grads}")
        print(f"   Target encoder grads: {target_encoder_grads} (should be 0 - EMA only)")
        
        assert online_encoder_grads > 0, "Online encoder should have gradients!"
        assert target_encoder_grads == 0, "Target encoder should NOT have gradients (EMA update)!"
        
        # Test EMA update
        print(f"\n✅ EMA update:")
        model.update_target_encoder(step=0, max_steps=1000)
        print(f"   Target encoder updated via EMA")
        
        print("\n✅ Pipeline complète fonctionne correctement!")
        
    finally:
        temp_path.unlink()


@DEPRECATED_MASKED_API
def test_pretrain_finetune_mode():
    """Test du switch entre mode pretrain et finetune."""
    print("\n" + "="*80)
    print("TEST 5: Pretrain vs Finetune Mode")
    print("="*80)
    
    model = create_jepa_tst_tiny()
    
    # Créer des données dummy
    B, T, C = 4, 128, 1
    x = torch.randn(B, T, C)
    
    # Test mode pretrain
    model.set_pretrain_mode(True)
    assert model.is_pretrain_mode() == True
    print("✅ Pretrain mode activated")
    
    num_patches = (T - model.patch_size) // model.stride + 1
    context_mask = torch.ones(B, num_patches, dtype=torch.bool)
    context_mask[:, num_patches//2:] = False
    target_mask = ~context_mask
    
    output_pretrain = model(x, context_mask=context_mask, target_mask=target_mask)
    print(f"   Pretrain output keys: {list(output_pretrain.keys())}")
    assert 'predictions' in output_pretrain
    assert 'targets' in output_pretrain
    
    # Test mode finetune
    model.set_pretrain_mode(False)
    assert model.is_pretrain_mode() == False
    print("\n✅ Finetune mode activated")
    
    output_finetune = model(x)
    print(f"   Finetune output keys: {list(output_finetune.keys())}")
    assert 'forecast' in output_finetune
    print(f"   Forecast shape: {output_finetune['forecast'].shape}")
    
    print("\n✅ Mode switching works correctly!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TEST COMPLET DE LA PIPELINE JEPATST")
    print("="*80)
    
    test_normalizer_identity()
    test_dataset_with_normalizers()
    test_datamodule_normalizer_types()
    test_full_pipeline_with_model()
    test_pretrain_finetune_mode()
    
    print("\n" + "="*80)
    print("✅ TOUS LES TESTS PASSENT!")
    print("="*80)
    print("\nRésumé:")
    print("  1. ✅ Identity normalizer ne modifie pas les données")
    print("  2. ✅ Dataset utilise le bon normalizer")
    print("  3. ✅ DataModule propage normalizer_type correctement")
    print("  4. ✅ Forward/backward pass fonctionnent avec JEPATST")
    print("  5. ✅ Switch pretrain/finetune mode fonctionne")
    print("\n🎉 Pipeline JEPATST prête pour l'entraînement avec RevIN!\n")