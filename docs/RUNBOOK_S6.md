# Runbook S4-c → critic (2026-09-07)

Toutes les commandes depuis la racine du repo, sur le pod, après `git pull`.
`PT` = le pretrain mini v3, `logs/` est ignoré par git.

```bash
mkdir -p logs
PT='checkpoints/timejepa_lotsa_mini_v3/pretrain_True/epoch00_valloss0.5495.ckpt'
```

Références appariées (head8 champion, flip + backtest) : 15 % 0.7974 / 0.5466, 25 % 0.7914 / 0.5433.
Stack officiel du champion (flip + mix + pool) : 0.7842 / 0.5340.

## 0. Rangement de l'éval scratch (jamais supprimer)

```bash
mkdir -p evaluation/timejepa_lotsa_mini_v3_head8_scratch_zs
mv evaluation/timejepa_lotsa_mini_v3_head8_zs/epoch00_valloss0.6565 \
   evaluation/timejepa_lotsa_mini_v3_head8_scratch_zs/
```

## 1. S4-c : contextes courts (une variable : la grille gagne 32 et 64)

```bash
PYTHONUNBUFFERED=1 python scripts/train.py --config-name lotsa_mini_v3_head8_ctx_zeroshot \
  "+training.pretrained_encoder_path=\"$PT\"" 2>&1 | tee logs/train_head8_ctx.log
```

Témoin dans le premier décile : les longueurs 32 et 64 apparaissent.

```bash
grep -o "context_len[^ ]*" logs/train_head8_ctx.log | sort | uniq -c | head
```

Éval à 15 % (remplacer `<CKPT>` par le checkpoint sauvé, `checkpoints/timejepa_lotsa_mini_v3_head8_ctx_zs/pretrain_False/epoch00_valloss*.ckpt`) :

```bash
CK=checkpoints/timejepa_lotsa_mini_v3_head8_ctx_zs/pretrain_False/<CKPT>
PYTHONUNBUFFERED=1 python scripts/evaluate_gift.py --config-name lotsa_mini_v3_head8_ctx_eval \
  +checkpoint_path=$CK +tta_flip=true +ratein=backtest 2>&1 | tee logs/eval_ctx_15_bt.log
```

Verdict P-ctx (2026-09-07, 15 %) : 0.8015 / 0.5521 contre 0.5466 apparié, 1/6 configs courtes en
baisse : ÉCHEC-DIAGNOSTIC, le régime court n'est pas un problème de données. **Le critic part de
head8 (section 2, variante a).** La variante b reste documentée mais n'est pas la voie.

## 2. Bras critic (S6, route A, α 0.05, N tiré dans {0,1,2,4,8}, cible jointe EMA)

Variante a, base head8 :

```bash
PYTHONUNBUFFERED=1 python scripts/train.py --config-name lotsa_mini_v3_head8_critic_zeroshot \
  "+training.pretrained_encoder_path=\"$PT\"" 2>&1 | tee logs/train_head8_critic.log
```

Variante b, base S4-c (même run, la grille de contextes en plus, déclarée comme bundle) :

```bash
PYTHONUNBUFFERED=1 python scripts/train.py --config-name lotsa_mini_v3_head8_critic_zeroshot \
  "+training.pretrained_encoder_path=\"$PT\"" \
  'training.context_lengths=[32,64,128,192,256,384,512,640,768,1024]' \
  2>&1 | tee logs/train_head8_critic_ctx.log
```

Le premier signal, dès les premières heures, sur wandb ou dans le log :

```bash
grep -o "critic/pinball_[0-8][^,]*" logs/train_head8_critic.log | tail -20
grep -o "val_critic/pinball_[0-8][^,]*" logs/train_head8_critic.log | tail -10
```

Les `pinball_i` absolues sont dominées par quelques items extrêmes du quart de batch : lire
`critic/pinball_gain` (= pinball_0 − pinball_N) et `critic/pinball_rel_gain`, lissés sur 50 points
(sur un run lancé avant ce témoin : expression wandb `${critic/pinball_0} - ${critic/pinball_N}`).
Il faut ce gain positif et croissant, `critic/energy_drop > 0`, `critic/delta_clipped_frac ≈ 0`.
Si `val_critic/pinball_8 ≥ val_critic/pinball_0` après un décile complet, le juge ne lit rien : couper.

Variante B (2026-09-08, une variable contre A : le prédicteur reçoit le gradient des pas
raffinés via z_pred dans E ; P-S6.3 au registre) :

```bash
PYTHONUNBUFFERED=1 python scripts/train.py --config-name lotsa_mini_v3_head8_critic_zeroshot \
  "+training.pretrained_encoder_path=\"$PT\"" \
  training.loss.critic_route=B model.name=timejepa_lotsa_mini_v3_head8_criticB_zs \
  2>&1 | tee logs/train_head8_criticB.log
# éval : même config lotsa_mini_v3_head8_critic_eval avec model.name=timejepa_lotsa_mini_v3_head8_criticB_zs
```

Bras λ_joint 0.1 et bundle B + λ 0.1 (2026-09-08, P-S6.4 / P-S6.5 au registre) :

```bash
C="python scripts/train.py --config-name lotsa_mini_v3_head8_critic_zeroshot +training.pretrained_encoder_path=\"$PT\""
PYTHONUNBUFFERED=1 $C training.loss.lambda_joint=0.1 model.name=timejepa_lotsa_mini_v3_head8_criticA-l01_zs \
  2>&1 | tee logs/train_head8_criticA_l01.log
PYTHONUNBUFFERED=1 $C training.loss.critic_route=B training.loss.lambda_joint=0.1 model.name=timejepa_lotsa_mini_v3_head8_criticB-l01_zs \
  2>&1 | tee logs/train_head8_criticB_l01.log
```

Lecture à 5k steps : les courbes `pinball_0 − pinball_N` lissées superposées ; A = 0.0005 en baisse.

## 3. Évals du checkpoint critic (à 15 % puis au meilleur)

`CK` = son checkpoint, `checkpoints/timejepa_lotsa_mini_v3_head8_critic_zs/pretrain_False/...`.
Toujours publier nu, flip, stack. Le raffinement se compare à SON mix-pool et à SON plafond.

```bash
CK=checkpoints/timejepa_lotsa_mini_v3_head8_critic_zs/pretrain_False/<CKPT>
E="python scripts/evaluate_gift.py --config-name lotsa_mini_v3_head8_critic_eval +checkpoint_path=$CK"

# officiel, sans boucle : nu, flip, stack
$E                                                  2>&1 | tee logs/eval_critic_nu.log
$E +tta_flip=true                                   2>&1 | tee logs/eval_critic_flip.log
$E +tta_flip=true +ratein=mix +ratein_pool=true     2>&1 | tee logs/eval_critic_stack.log

# officiel, AVEC la boucle (le seul chiffre de S6) : boîte 0.4, même que le plafond
$E +tta_flip=true +ratein=mix +ratein_pool=true +refine=energy +refine_alpha=0.05 \
                                                    2>&1 | tee logs/eval_critic_stack_refine.log

# diagnostic, jamais officiel : le plafond de CE checkpoint, pour la récupération g/G
$E +tta_flip=true +ratein=mix +ratein_pool=true +refine=ceiling +refine_alpha=0.05 \
                                                    2>&1 | tee logs/eval_critic_stack_ceiling.log
```

Récupération = (stack − stack_refine) / (stack − stack_ceiling). P-S6.1 : 10-25 %. Prédiction
utilisateur : 0.49 de CRPS, soit ≈ 26 %.

## 4. Ablation conditionnelle : bras joint (N = 0)

Seulement si le critic gagne (prouver que c'est la boucle) ou perd (savoir si le terme seul est
neutre). Même variante de base que le critic.

```bash
PYTHONUNBUFFERED=1 python scripts/train.py --config-name lotsa_mini_v3_head8_joint_zeroshot \
  "+training.pretrained_encoder_path=\"$PT\"" 2>&1 | tee logs/train_head8_joint.log
# éval : lotsa_mini_v3_head8_joint_eval, nu / flip / stack comme en 3
```

## Digest à m'envoyer

```bash
grep -h "vs_official\|REFINE\[\|coverage" logs/eval_*.log
```
