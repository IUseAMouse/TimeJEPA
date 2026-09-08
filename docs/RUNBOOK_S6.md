# Runbook S4-c → critic → score matching (2026-09-07 / 08)

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

## 5. Bras S6-b : score matching (2026-09-08, remplace la voie critic, close par diagnostic)

Une variable contre le bras joint : le terme de score (λ 0.1, route B, perturbations niveau /
bruit / pente / médiane du fan, moitié du batch). Boucle critic OFF. Base head8 + pretrain mini.

```bash
PYTHONUNBUFFERED=1 python scripts/train.py --config-name lotsa_mini_v3_head8_score_zeroshot \
  "+training.pretrained_encoder_path=\"$PT\"" 2>&1 | tee logs/train_head8_score.log
grep -n "H2b/S6 settings" logs/train_head8_score.log      # audit des clés au démarrage
```

Témoins wandb, dans l'ordre où ils tombent :
- `score/cos` et surtout `score/cos_level` (l'axe du plafond) qui montent vers 1 dès les premiers
  milliers de steps ; `score/cos_forecast` = la direction depuis le fan réel ;
- premier point de validation : **`val_score/valley_frac` > 0.8** (la sonde, interne), `val_loss`
  contre head8 au même step (coût sur le fan nu, P-S6b.4) ;
- débit it/s (attendu ≈ ×1.5 contre le finetune plain).

Réception sur le checkpoint 5 % (`CK`), sans GPU d'entraînement :

```bash
CK=checkpoints/timejepa_lotsa_mini_v3_head8_score_zs/pretrain_False/<CKPT>
P="python scripts/probe_energy_shift.py --checkpoint $CK --model-config lotsa_mini_v3_head8_score_eval \
   --configs m_dense/D/short,loop_seattle/H/short,m_dense/H/short,electricity/H/short,solar/H/short --instances 64"
$P --center truth --out logs/probe_score_truth.json 2>&1 | tee logs/probe_score_truth.log   # P-S6b.1 : vallée en 0 > 80 %
$P --center fan   --out logs/probe_score_fan.json   2>&1 | tee logs/probe_score_fan.log     # P-S6b.2 : signe > 0.7
```

Puis, au 15 %, les évals (même schéma que §3, config `lotsa_mini_v3_head8_score_eval`) : nu, flip,
stack, stack + `+refine=energy +refine_alpha=0.05` (le chiffre de S6-b, P-S6b.3), stack +
`+refine=ceiling +refine_alpha=0.05` (son plafond). Récupération = (stack − refine) / (stack − ceiling).

## 6. BiasIN : correction causale du biais de niveau (2026-09-08, éval seule, aucun GPU d'entraînement)

Le plafond de raffinement mélangeait biais systématique et bruit réalisé. BiasIN ne garde que la
part qui PERSISTE : biais du centre mesuré sur les fenêtres de backtest de RateIN (k = 1, même
TTA), en unités d'échelle du contexte ; rétrécissement λ ∈ {0.25, 0.5, 1} validé par config en
appliquant le biais de la fenêtre ancienne à la récente (ratio de pinball poolé < 0.95, sinon
no-op). Couche finale sur le fan natif, après mix. `+bias=oracle` = décalage constant par instance
lu sur la cible : borne du biais de niveau systématique, diagnostic, jamais officiel.

```bash
CK=checkpoints/timejepa_lotsa_mini_v3_head8_zs/pretrain_False/epoch00_valloss0.6522.ckpt
E="python scripts/evaluate_gift.py --config-name lotsa_mini_v3_head8_eval +checkpoint_path=$CK +tta_flip=true +ratein=mix +ratein_pool=true"
PYTHONUNBUFFERED=1 $E +bias=oracle    2>&1 | tee logs/eval_head8_stack_bias_oracle.log   # borne (diagnostic)
PYTHONUNBUFFERED=1 $E +bias=backtest  2>&1 | tee logs/eval_head8_stack_bias_bt.log       # officiel
grep -h "vs_official\|BIAS\[\|coverage" logs/eval_head8_stack_bias_*.log
```

Référence : stack 0.7842 / 0.5340. Dossiers `gift_flip_ratein-mix-pool_bias-oracle` et `..._bias-bt`.
Lecture : `configs active`, `lambda hist`, `mean |beta|` dans le bloc BIAS ; par config `lambda`,
`n_val`, `|shift|`. P-BI.1 / P-BI.2 au registre.

## 7. ANNEAL-30 : finetune head8 recuit à 30 % de l'époque (2026-09-08)

```bash
PYTHONUNBUFFERED=1 python scripts/train.py --config-name lotsa_mini_v3_head8_anneal30_zeroshot \
  "+training.pretrained_encoder_path=\"$PT\"" 2>&1 | tee logs/train_head8_anneal30.log
grep -n "schedule_fraction" logs/train_head8_anneal30.log     # « annealed and run bounded at 30% »
```

Témoin wandb : `lr-AdamW` (ou `lr-*`) descend vers `min_lr` en fin de run (~22 h à 8 it/s).
Éval du DERNIER checkpoint puis des 15 % et 25 % : `lotsa_mini_v3_head8_eval`, flip + backtest,
puis stack (`+ratein=mix +ratein_pool=true`). P-ann.1 / P-ann.2 au registre ; référence
apparié 15 % 0.5466, 25 % 0.5433, stack 0.5340.

## Digest à m'envoyer

```bash
grep -h "vs_official\|REFINE\[\|coverage" logs/eval_*.log
```
