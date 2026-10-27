# RL Controller — thiết kế và phân tích hiện tại

Tài liệu chính của nhánh adaptive latent handoff. Cập nhật kết quả đến
27/10/2026,20:10 UTC+7 sau khi đối chiếu raw artifacts trên server.
Đọc [§5 — kết quả](#verified-results),
[§6.1 — vấn đề dữ liệu](#train-data-priority) và §10 — trạng thái job trước
khi đề xuất thí nghiệm. Protocol controller lịch sử được giữ để giải thích
các run đã chạy, không phải lệnh tự động tiếp tục các nhánh đã dừng.

<a id="native-kv-contract"></a>

## 0. Backend bắt buộc cho run mới: HF native + direct KV

**Quyết định người dùng ngày 27/10/2026: mặc định mọi lần profile, rollout lấy
reward, validation và test LatentMAS mới phải dùng HF native, truyền trực tiếp
KV từ A sang B.** Quy định này ưu tiên hơn các ví dụ/config hybrid lịch sử bên dưới.

Luồng mặc định:

~~~text
A prompt → HF prefill → K latent steps → full KV của A
                                         ↓ past_key_values
                              B xử lý prompt mới → HF decode đáp án
~~~

Realignment ánh xạ hidden sang latent input bên trong A. Truyền A→B dùng KV
layer-wise chứa prompt A và latent context; không dựng lại các embedding của A
để B prefill lần nữa. B vẫn cần tính prompt của chính B. Mask padding, causal
mask và vị trí RoPE phải nối tiếp cache đúng theo từng câu.

Phải đọc entrypoint thực tế và call site sinh đáp án B trước khi chạy.
Đường HF methods/latent_mas.py::run_batch() có truyền past_key_values;
đường run_batch_vllm() và nhiều script controller lịch sử dựng prompt_embeds
rồi gọi vllm_engine.generate. Chỉ bỏ flag --use_vllm hoặc đổi config chưa
đảm bảo script đã native: có entrypoint hardcode use_vllm=True.
Nếu script chưa hỗ trợ native thì cần triển khai/xác minh đường native trước,
không chạy hybrid rồi đặt tên native. Đây là hướng dẫn cho run mới, chưa phải
tuyên bố mọi entrypoint/controller hiện có đã chuyển backend.

| Backend / handoff cần ghi rõ | Ý nghĩa |
|---|---|
| native_hf / direct_full_kv | Mặc định: B kế thừa full KV của A |
| hf_vllm / embedding_replay | A chạy HF, B prefill lại embedding prefix bằng vLLM; thí nghiệm riêng |
| native_hf / embedding_replay | Đối chứng replay trên HF; vẫn không phải direct KV |
| native_hf / selective_kv | Chỉ giữ đầy đủ prefix ở layer được chọn; thay đổi phương pháp truyền |

Chỉ dùng hybrid/replay hoặc selective KV khi người dùng yêu cầu rõ thí nghiệm
đó hoặc phép đối chứng tương ứng. Thiếu VRAM thì chờ tài nguyên, không tự đổi
backend/handoff. Job KVComm ba nhánh đã được cho phép giữ nguyên protocol
đóng băng; đối chứng chính cho selective KV là full KV native.

Trước lượt GPU dài cần kiểm tra nhỏ trên model thực: KV kế thừa giữ đúng giá trị,
mask/position đúng, logits hoặc generation đối chiếu native-reference.
Manifest/runtime phải lưu entrypoint, revision model, backend, handoff,
decoder, batch/cache protocol và bằng chứng kiểm tra. use_vllm=false tự nó
không đủ chứng minh native; prefix caching của vLLM không nhập cache của HF.

**Đính chính lịch sử:** profile train256/valid128, train1024 expansion, các run
controller PPO/GRPO, supervised/full-grid, EMA K10–80 và SRR A→B đã audit chủ yếu
dùng hybrid embedding replay. Fixed K10 69,6%, supervised 70,4% và nhóm
adaptive-help 99/1024 thuộc protocol đó. KV debate 3 round dùng direct native KV;
single-model SRR giữ cache trong cùng model, không có handoff A→B.
Không đổi nhãn raw results hoặc coi checkpoint/quality labels hybrid là đã được
xác nhận trên native. Nếu nghiên cứu tái sử dụng qua backend, phải ghi rõ và
đánh giá lại. Các run thất bại/chưa xong không phải kết quả chất lượng.

Nguồn đối chiếu: [paper, Figure 3 và implementation](https://arxiv.org/pdf/2511.20639#page=4),
[upstream HF execution](https://github.com/Gen-Verse/LatentMAS/blob/main/methods/latent_mas.py),
[upstream backend guidance](https://github.com/Gen-Verse/LatentMAS#-latentmas-with-vllm).
Bằng chứng từng run giữ ngoài docs tại
/data/nghialt/.remote-workspace/jobs/latentmas-backend-history-audit-20261026-v1/.

## 1. Bài toán và phạm vi

Hai agent Qwen3-4B được giữ cố định: A tạo latent trajectory, B nhận prefix để
sinh đáp án. Controller học chọn thời điểm chuyển giao theo từng câu, nhằm
tăng strict correctness của B. Chưa có kết luận adaptive tốt hơn fixed budget.

- A tạo latent input e_t từ h_(t-1) qua realignment rồi forward để nhận h_t.
- h_0 là last-layer hidden tại token thật cuối của prompt sau prefill.
- Tại checkpoint k, current hidden là h_k; previous checkpoint là h_(k-10).
- A tích lũy context của latent inputs e_1…e_k trong KV; không đồng nhất e_t với h_t.
- Run mới chuyển KV này trực tiếp sang B theo §0. Các run hybrid lịch sử chuyển
  embedding prefix rồi B prefill lại; phải giữ riêng nhãn protocol.
- Các run controller lịch sử chỉ cập nhật controller, không fine-tune Qwen.
  Shortcut adapter ở §10 là hướng riêng, chưa có kết quả training.

## 2. State controller cosine đã chạy: ba nhóm

Schema: hk_cosine10_k_train_zscore_v1. Tổng cộng 2.562 chiều.

| Nhóm | Giá trị | Xử lý trong controller |
|---|---|---|
| Hidden của A | h_k, 2,560 chiều | LayerNorm riêng theo observation, epsilon 1e-5 |
| Thay đổi trong 10 bước | d_k = clamp(1 − cos(h_k, h_(k−10)), 0, 2) | Tính FP32, cosine epsilon 1e-8; z-score bằng train-only mean/std |
| Tiến độ | k/80 | Giữ nguyên |

State causal: chỉ dùng hiện tại và quá khứ. Không đưa gold, kết quả B, future
hidden, timing, token count hoặc availability masks vào policy. Không thêm
patience/probe hoặc đổi sang cosine giữa hai bước liền nhau.

Thống kê scalar chỉ fit một lần trên train1024 × 7 checkpoints và lưu vào
checkpoint; std dưới epsilon dùng scale 1. Không fit lại ở validation/test.
Normalization chỉ áp dụng cho input controller, không sửa A, realignment hay B.

Kiến trúc: Linear(2562 → 128), LayerNorm(128) không affine, GELU,
actor 2 logits và critic 1 giá trị. Hidden LayerNorm cũng không affine.

[State cosine](../controller/cosine_state.py),
[phần dùng chung](../controller/three_group.py),
[checkpoint loader](../controller/checkpoints.py).

L2 là schema đối chứng cũ; giữ loader để đọc checkpoint, không coi đó là hướng
train mặc định. Hai state cùng dimension nhưng khác schema và scalar buffers,
không được dùng checkpoint thay thế lẫn nhau.

## 3. Action, reward và loss

| Thành phần | Quy định |
|---|---|
| Action | HANDOFF=0; CONTINUE=1, tiếp tục thêm 10 latent steps |
| Checkpoints | K10,20,…,70 có quyết định; K80 bắt buộc handoff, không có learned action |
| Reward | 1 nếu đáp án B đúng theo strict scorer, ngược lại 0 |
| Discount | gamma=1; terminal reward được dùng làm Monte Carlo return cho các quyết định đã đi qua |
| PPO | clip=0,2, value weight=0,5, entropy weight=0,01 |
| Loại bỏ khỏi objective hiện tại | time/step penalty, tie auxiliary, KL anchor và supervised warm-start |

PPO loss gồm clipped actor surrogate, value MSE và entropy bonus:

~~~text
L = L_clipped_actor + 0.5 * L_value_MSE - 0.01 * H(policy)
advantage_t = terminal_reward - old_value_t
~~~

Actor loss cộng các quyết định đã visited của từng episode rồi lấy trung bình
theo episode; value/entropy lấy trung bình trên visited decisions. Padding chỉ
phục vụ bookkeeping của loss, không phải policy input.

[PPO math](../controller/ppo.py),
[optimizer updates và logging](../controller/training.py),
[live runtime](../controller/online_three_group.py).

## 4. Protocol PPO lịch sử và điều kiện áp dụng cho run mới

Các thông số bên dưới mô tả campaign fresh-rollout PPO đã chạy bằng hybrid.
Chúng không phải lệnh mặc định để tái chạy nguyên xi trên HF native.
Run mới phải đáp ứng §0 và có entrypoint native đã kiểm tra trước.

Đây là online PPO với rollout mới, không phải tối ưu bảng grid.

- Train pool 1.024 câu official train; validation 128 câu; test 500 câu tách riêng.
  Khóa source IDs/hash, kiểm overlap theo câu; các seed/K của cùng câu không phải
  các mẫu độc lập để chia fold.
- Mỗi iteration chọn 32 câu bằng shuffled sampler, chạy A và gọi B mới tại K
  policy chọn. Hai cohort 16 thu reward trước khi update.
- Sau đó chạy 4 optimizer epochs, minibatch 8: 16 optimizer updates/iteration.
  Không gọi lại B cho từng optimizer update.
- Stochastic actions khi thu train rollout; greedy khi validation/test.
- Lịch pilot cosine: 64 iterations, 2.048 train B completions, hai lượt qua pool.
  Validation128 tại iterations 0/16/32/40/48/64: 768 validation B completions.
- LR 3e-4; seed42; khởi tạo mới, không lấy PPO weights cũ.
  Lịch 64 là pilot để kiểm tra, chưa phải số vòng tối ưu.
- Profile features dùng để fit normalization và kiểm tra dữ liệu; correctness
  trong profile không thay terminal rewards từ B mới.

Offline trong tên config chỉ là W&B offline storage, không có nghĩa train offline.
Hướng grid objective đã rời khỏi luồng làm việc hiện tại; không tự chuyển lại.

[Sampler](../controller/sampler.py),
entrypoint server-only scripts/train_cosine_state_ppo.py,
[template lịch sử](../configs/controller_ppo_cosine_state64_offline_v2.yaml).
Run hoàn tất mới hơn là cosine v8; xem §5.2.

<a id="verified-results"></a>

## 5. Kết quả đã xác minh — cập nhật 27/10/2026, 20:10 UTC+7

Các bảng test500 dưới ghi run hoàn tất trên cùng bộ MedQA500 đã dùng nhiều
lần; đây không còn là test chưa từng xem. Pilot Claude192 câu được tách ở §5.5.
Không gộp hybrid embedding replay, native direct KV và single-model thành
một bảng xếp hạng chung. Checkpoint controller được chọn bằng validation.

### 5.1. Native HF: Full KV và selective KVComm, A K10

Cùng Qwen3-4B, batch4, seed42, decoder và cohort. Cả ba nhánh chạy HF native;
embedding replay ở đây cũng chạy HF, khác các baseline HF+vLLM lịch sử.

| Handoff | Đúng /500 | Accuracy | Loop | Parse fail | Thời gian đo (s) |
|---|---:|---:|---:|---:|---:|
| Embedding replay | 349 | 69,8% | 1 | 7 | 18931 |
| Direct Full KV | 341 | 68,2% | 1 | 6 | 17099 |
| Selective KVComm | 355 | 71,0% | 1 | 11 | 19622 |

Selective so với Full KV: +2,8 điểm %, paired question-bootstrap 95% CI
[-0,4; 6,0] điểm %, cứu 40/hại 26. Chưa đủ bằng chứng tăng accuracy ổn định.
Giảm 69,28% số byte prefix KV được truyền, nhưng thời gian đo tăng 14,76%;
nếu cộng calibration layer được phân bổ, selective là 19699 giây.
Đây là phép đo cùng thiết bị, không phải latency truyền KV qua mạng.

Ở Full KV, A chiếm 0,49% và B decode chiếm 99,35% thời gian đo. Giảm K của A
đơn thuần khó tạo speedup lớn trong cấu hình này. Tỷ lệ thời gian trên là
tổng các phần đo theo protocol; không dùng để suy latency cá nhân hay kết quả
multi-GPU đang chờ.

Nguồn server: evaluations/medqa500_latentmas_kvcomm_k10_20261026_v1,
verified_results.json và comparison.json. Job:
latentmas-kvcomm-k10-test500-20261026-v1.

### 5.2. Controller trên protocol hybrid lịch sử

| Phương pháp / checkpoint validation | Đúng /500 | Accuracy | Mean K |
|---|---:|---:|---:|
| Fixed K10 lịch sử | 348 | 69,6% | 10,00 |
| PPO L2 best40 | 344 | 68,8% | 31,80 |
| PPO L2 last200 | 339 | 67,8% | 22,82 |
| PPO cosine64 best32 | 348 | 69,6% | 23,22 |
| GRPO cosine64 best48 | 343 | 68,6% | 34,40 |
| GRPO C1 late fusion | 350 | 70,0% | 10,38 |
| GRPO C2 GRU | 348 | 69,6% | 10,60 |
| Supervised first-success C1 | 352 | 70,4% | 10,12 |
| Full-grid quality C1 | 343 | 68,6% | 36,00 |
| Live-quality GRPO C1 | 326 | 65,2% | 25,12 |
| Warm50 + adaptive-help GRPO | 323 | 64,6% | 73,42 |

Cosine v2 là attempt lỗi khởi tạo, không phải run mới nhất: v8 đã hoàn tất 64
iterations và test500 dùng best validation iteration 32. Không lấy lỗi OOM/vLLM
của retry trước làm kết luận về chất lượng controller.

Supervised đạt 70,4% nhưng 497/500 câu vẫn K10; ba câu đổi K cứu 0/hại 0 so với
fixed K10. Chênh lệch tổng +4 câu nằm trong nhóm giữ K10, nên chưa chứng minh
lợi ích chọn K. C1 chỉ đổi K ở 5 câu, cứu 2/hại 1; cần xem cả drift ở nhóm giữ K10.
GRPO thường có 85–87% nhóm 4 rollout cùng reward; warm/mixed giảm còn 64,84%
nhưng test giảm còn 64,6%. Loss giảm hoặc reward contrast tăng chưa đủ.

### 5.3. EMA beta=0,05: đủ K10–80, hybrid embedding replay

| K | Realignment lịch sử | EMA beta=0,05 | Loop EMA /500 |
|---:|---:|---:|---:|
| 10 | 69,6% | 68,4% | 0 |
| 20 | 69,0% | 68,4% | 0 |
| 30 | 67,2% | 68,6% | 0 |
| 40 | 66,0% | 67,8% | 1 |
| 50 | 63,4% | 66,6% | 3 |
| 60 | 61,6% | 67,6% | 1 |
| 70 | 64,2% | 66,6% | 3 |
| 80 | 63,6% | 67,2% | 1 |

Cột Realignment là grid lịch sử, không phải tám đối chứng chạy mới cùng lúc.
Ở phép so fresh matched K40, Realignment 327/500=65,4%, EMA 339/500=67,8%;
loop 20→1, parse fail 25→2; accuracy delta+2,4 điểm %, CI95%[-0,8;5,8].
Có tín hiệu giảm degeneration nhưng chưa xác nhận tăng accuracy ổn định.

Trong EMA, so K10 với K40 có 25 câu K10 sai→K40 đúng, 28 câu đúng→sai:
không thể chỉ học nhóm được cứu mà bỏ qua harm. Đây là phân tích cùng grid
hybrid; chưa có kết quả EMA Full KV native đủ các K.
Nguồn: evaluations/medqa500_ema_k10_k80_20261023_v1 và
medqa500_ema_k10_k40_20261023_v2.

### 5.4. SRR, debate và dừng sớm

| Thí nghiệm hoàn tất | Protocol | Đúng /500 | Accuracy |
|---|---|---:|---:|
| Realignment K2 | A→B hybrid | 347 | 69,4% |
| Realignment K10, run so projection | A→B hybrid | 349 | 69,8% |
| SRR độc lập, 2 refinement một lần; không chuỗi K | A→B hybrid | 341 | 68,2% |
| Single-model SRR, 2 refinement | Native HF, một model | 322 | 64,4% |
| Debate 3 round; K10/lượt A và B, B đáp cuối | Native direct KV | 343 | 68,6% |

Prompt/backend/batch của single-model và các run A→B khác nhau; bảng này mô tả
kết quả từng protocol, không phải ablation chỉ đổi một yếu tố. Bản SRR cũ gắn
refinement vào chuỗi K10 không phải SRR độc lập được người dùng chốt sau đó.

Stopping hybrid với cap1024 và forced final answer: fixed EMA K10 = 67,2%;
cosine+entropy = 66,4% (498/500 câu đi K80); cosine-only = 65,4%,
A thực chạy 42,442 bước trung bình rồi rollback về mean handoff K31,442.
Cả hai không đạt chứng nhận adaptive theo calibration đã đặt. Rollback không
hoàn lại compute đã thực hiện. Không so trực tiếp với bảng cap4096 phía trên.

Audit 6 loop cases và 14 controls của debate thấy cosine hidden B theo đúng
chu kỳ lặp đạt 0,985–0,998 trong lúc decode. Hidden latent trước decode chưa
cho tín hiệu rõ hơn controls; đây chưa phải detector dừng A được xác nhận.

### 5.5. Phần Claude: diagnostic và pilot, không phải test500 mới

- Greedy B repeat: 96/96 cặp có cùng token output và correctness.
  Chọn K theo greedy rồi đánh giá bằng sampled B chỉ tăng 69,01→69,27% trên
  train256 và 59,11→60,94% trên validation128 trong diagnostic lịch sử.
  Đây không phải controller tổng quát hóa sang câu mới.
- PCA32 + logistic probe trên noEMA profile: validation 59,11%, meanK10,
  bằng fixed K10; không có lợi ích adaptive được quan sát.
- Pilot MedQA train 192 câu ×2 decoder seeds, full generation:

| Cấu hình Claude | Accuracy /384 completions |
|---|---:|
| No latent, nhãn none-1 trong pilot | 69,27% |
| Realignment K10 | 68,49% |
| EMA beta=0,05 K10 | 69,53% |
| EMA beta=0,05 K20 | 68,75% |
| EMA beta=0,05 K40 | 69,27% |

Hai seed của cùng câu không phải 384 câu độc lập. Pilot này còn có ARC/GSM8K
và các cap/no-thinking khác; không chọn cấu hình thắng rồi gọi là cải thiện
test500. No-latent của pilot dùng protocol riêng, chưa thay được matched
B-only / A-prompt-K0 baseline của native test500.

### 5.6. Insight có bằng chứng và giới hạn

1. Nhóm adaptive-help train chỉ 99/1024 theo tiêu chí đúng≥2/3 seed (§6.1).
2. Oracle nhìn đủ bảng quá lạc quan: chọn K bằng hai seed rồi kiểm seed thứ ba
   chỉ tăng 0,75 điểm % train, CI95%[-0,36;1,89], và 2,08 điểm % validation,
   CI[-1,04;5,73]. Diagnostic vẫn biết outcome của cùng câu ở seed khác;
   không phải causal policy hoặc bằng chứng tổng quát hóa.
3. Probe causal-state OOF AUROC cho CONTINUE tại K10 khoảng 0,486–0,502,
   gần permuted-label control 0,493. Chưa có bằng chứng state hiện tại dự đoán
   lợi ích downstream tốt trên câu mới.
4. Native selective KV có tín hiệu về nội dung truyền; EMA có tín hiệu ổn định
   trajectory. Cả hai cần phép so matched và đánh giá lặp lại, chưa có kết luận
   phương pháp mới tăng accuracy và giảm thời gian đồng thời.

Nguồn audit: /data/nghialt/.remote-workspace/jobs/latentmas-insight-audit-20261027-v1/
(analysis.json, source hashes, paired outcomes); phần Claude đối chiếu các job
claude-greedy-b-20261022-v1, claude-signal-audit-20261022-v1,
claude-online-20261023-v1 và probe-pca-logistic-20261025-v1.
Lượt cập nhật này kiểm lại 1947 checksum nguồn; không chạy thêm model.
Trạng thái chưa hoàn tất xem §10, không suy accuracy từ phần kết quả đang dở.

## 6. Phân tích còn cần theo dõi

<a id="train-data-priority"></a>

### 6.1. Ưu tiên cao: nhóm adaptive-help chỉ khoảng 10% train

**Đây là vấn đề dữ liệu trọng tâm cần phân tích trước khi đề xuất đổi state,
loss, kiến trúc hoặc train dài hơn. AI/người mới tiếp nhận cần đọc phần này và
xoáy sâu vào tín hiệu giúp chọn K, thay vì chỉ tối ưu đường train loss.**

Kiểm tra ngày 22/10/2026 trên profile đủ tám K10–80, ba seed 42/43/44:
một K “đạt” khi B trả lời đúng theo strict scorer ở ít nhất 2/3 seed.
Parse failure vẫn tính sai. Đây là định nghĩa phân nhóm trên bảng lịch sử,
không phải xác suất đúng thực tế đã được xác nhận bằng nhiều lần chạy mới.

| Nhóm câu, theo cùng ngưỡng 2/3 | Train (1,024 câu) | Validation (128 câu) |
|---|---:|---:|
| Đã đạt ở K10 | 704 (68,75%) | 75 (58,59%) |
| Chưa đạt ở K10, nhưng ít nhất một K20–80 đạt: **adaptive-help** | **99 (9,67%)** | **16 (12,50%)** |
| Không K nào đạt ngưỡng | 221 (21,58%) | 37 (28,91%) |

**Không viết “K10 chỉ sai khoảng 10%”.** Tổng nhóm chưa đạt ở K10 là
320/1.024 = 31,25%; trong đó 99 câu có K sau đạt ngưỡng, còn 221 câu chưa
có K nào đạt ngưỡng. “Không đạt 2/3” cũng không đồng nghĩa sai cả ba lần:
151/1.024 câu mới là nhóm sai ở mọi K trên cả ba seed.

Ý nghĩa đối với controller:

- Nhóm chuyển từ chưa đạt sang đạt là thiểu số. Accuracy/CE tổng thể có thể
  thiên về dừng ở K10 và che mất khả năng nhận diện các câu cần CONTINUE.
  Ví dụ nhãn supervised first-success hiện có 800 HANDOFF và 232 CONTINUE
  trên 1.032 quyết định được gán nhãn; K80 là forced handoff, không có nhãn
  hành động học được tại K80.
- Cần xét đồng thời câu được cứu và câu vốn đúng bị làm sai khi đi tiếp.
  Cải thiện trên 99 câu không tự bảo đảm cải thiện accuracy toàn bộ tập.
- 9,67% là quy mô nhóm theo tiêu chí trên, **không phải mức tăng accuracy
  hứa hẹn hoặc cận trên hiệu năng online**. Câu đạt 2/3 ở K10 vẫn có thể
  thay đổi tỷ lệ đúng ở K khác; phân nhóm này không mô tả hết các thay đổi đó.
- K “tốt nhất” được chọn sau khi xem đủ bảng có selection bias. Profile còn
  chứa các điều kiện GB10/H100 và engine lifetime khác nhau. Không kết luận
  K lớn giúp một câu một cách ổn định chỉ từ một seed hoặc bảng oracle.

**Các câu hỏi bắt buộc cho AI/người tiếp tục explore:**

1. **Lợi ích có lặp lại không?** Giữ K ứng viên đã chọn từ dữ liệu khám phá,
   kiểm K10 và K đó bằng seed mới, cùng hardware/prompt/decoder/batch/cache.
   Không chọn lại K tốt nhất trên chính các lần chạy dùng để kiểm chứng.
2. **State có nhận diện được nhóm cần đi tiếp không?** Dùng question-level
   folds, train-only normalization và các baseline đơn giản; báo precision,
   recall/PR-AUC cho CONTINUE cùng kết quả toàn tập. Không chia các K/seed
   của một câu sang cả train và held-out fold. Kết quả B/future K chỉ dùng
   tạo target train hoặc phân tích, tuyệt đối không đưa vào state khi chạy.
3. **Sampler/nhãn có làm mất nhóm thiểu số không?** Báo riêng ba nhóm trong
   bảng; kiểm tỷ lệ HANDOFF/CONTINUE, độ phủ câu/K và các nhóm reward bằng
   nhau của GRPO. Sampling cân bằng hoặc weighting là giả thuyết cần so sánh,
   không tự thay protocol hay chỉ đánh giá trên tập adaptive-help.
4. **Net gain có dương trên đủ câu không?** Báo accuracy, rescue/harm so với
   fixed K10, phân bố K và kết quả từng nhóm trên toàn validation/test, kể cả
   câu không có K đạt ngưỡng. Tách kết quả bảng cũ, accuracy nhãn hành động
   và correctness của B chạy mới; loss giảm không thay cho kiểm tra này.

Nguồn để kiểm lại và lấy source IDs của từng câu:

- Profile gốc: `profiles/medqa_train1024_valid128_h100_expansion_20261018_v1`.
- Bảng đầy đủ và kết quả từng câu:
  `/data/nghialt/.remote-workspace/jobs/latentmas-fullK-correct-exclusive-20261022-v1/summary.json`
  và `per_question.jsonl` trong cùng thư mục.
- Danh sách ba nhóm train/validation:
  `/data/nghialt/.remote-workspace/jobs/latentmas-warm-mixed-grpo64-20261022-v1/sampler_strata.json`.
- Nhãn supervised và provenance:
  `matched_runs/supervised_train1024_firstK_majority_c1_50_h100_20261022_v1/labels/`
  (resolve attempt từ `complete.json`, không sửa raw artifacts).

### 6.2. Các phân tích khác còn cần theo dõi

1. **Cosine chỉ là tín hiệu hình học.** Hidden ít đổi chưa chứng minh B đã có
   đáp án đúng. Thay L2 bằng cosine cần được kiểm tra bằng live B.
2. **State cũ đã normalize.** Không diễn giải vấn đề trước đây là quên normalize.
   LayerNorm/z-score cũng không bảo đảm ba nhóm đóng góp ngang nhau.
3. **Train stochastic, deploy greedy.** Theo dõi stop probabilities, reached
   counts, entropy và phân bố K; threshold argmax có thể tạo phân bố rất khác
   stochastic rollout. Không đánh giá chỉ bằng meanK hoặc train loss.
4. **Reward có nhiễu theo điều kiện chạy.** Batch/padding, hardware, kernel,
   grouping và cache có thể đổi trajectory hoặc B output. So sánh paired trên
   cùng câu và protocol; không quy mọi thay đổi validation cho PPO.
5. **Parse/cap cần tách riêng.** length finish là output token cap, chưa có nghĩa
   tràn context. Parse failure tính sai theo scorer đã khóa; không sửa scoring
   theo gold. Thêm yêu cầu B chốt đáp án sẽ là thay prompt/protocol, cần so sánh
   riêng, không trộn với ablation state.
6. **Oracle khác controller.** Any-K/any-seed đúng là retrospective upper bound
   có sử dụng outcomes; không báo như accuracy khả thi đã đạt của một run adaptive.

Ưu tiên nghiên cứu tiếp theo là kiểm chứng vấn đề dữ liệu và khả năng nhận diện
nhóm adaptive-help ở §6.1 trước khi kết luận cần đổi state/loss/kiến trúc.
Các kiểm tra này là hướng phân tích, không tự cấp quyền mở thêm ablation,
đổi protocol hoặc kéo dài training ngoài các run người dùng đã yêu cầu.

## 7. Evaluation và logging

Chọn checkpoint bằng validation accuracy; hòa thì lower meanK, rồi iteration
sớm hơn. Khóa checkpoint trước test. Không chọn lại theo test accuracy.

Run mới mặc định HF native + direct KV theo §0. Giữ Qwen3-4B BF16,
pinned revision, prompt, scorer và decoder theo protocol đã khóa; ghi backend/
handoff trong manifest và bảng kết quả. Không đổi decoder cùng lúc với ablation
truyền thông tin nếu cần đo riêng tác động của handoff.

Protocol hybrid lịch sử dùng B temperature0.6/top_p0.95/top_k0/cap4096/batch16;
các run stopping riêng đã dùng cap1024 + ép chốt đáp án. Không trộn các nhóm đó
thành một baseline native. Batch native phải được ghi rõ và kiểm tra bộ nhớ,
không tự suy ra batch16 luôn phù hợp. Manual think=false không đồng nghĩa
chat-template enable_thinking=false. Native handoff phải giữ đúng KV, padding/
causal masks và logical positions; ragged prompt embeddings thuộc đường replay.

MedQA primary scorer là medqa_boxed_v1: boxed cuối, strip/case-normalize,
chấp nhận A/B/C/D; parse failure là sai. Log raw output/token IDs, finish reason,
prompt hash và config để truy vết; không tự đổi parser hoặc prompt trong so sánh.

Theo dõi accuracy, meanK và K counts, rescue/harm paired với fixed, parse/cap/loop,
loss policy/value/entropy, KL/clip/gradient norm, critic và train coverage.
Timing là report-only; meanK nhỏ hơn chưa chứng minh latency thấp hơn.

W&B project: theanha2hvtls-ai4life/Controller-Latent-Reasoning.
[Tracker](../controller/tracking.py) và [scalar mirror](../controller/tracking_mirror.py)
ghi riêng optimizer_step/iteration; giữ log local khi upload lỗi. Không upload
prompts/gold/checkpoints mặc định.

Job launch, model loading, initial validation và committed PPO update là các
trạng thái khác nhau. Chỉ báo train hoàn tất khi đủ committed iterations,
training_finished.json và exit code 0; tmux tồn tại/biến mất không chứng minh thành công.

## 8. Vận hành và phạm vi bản GitHub

Workspace /data/nghialt; repo /data/nghialt/LatentMAS; SSH alias brev-h100.
Runtime Conda latentmas, prefix /data/nvidia/conda_envs/latentmas.
Private tmux socket: /data/nghialt/.remote-workspace/tmux.sock.
Không thao tác process của người khác; kiểm identity/start time/task provenance.

Theo yêu cầu người dùng, scripts/ và tests/ đã bỏ khỏi nhánh main GitHub tại
commit 473cb2e và được ignore. Các file vẫn có trên server; không đưa lại lên
Git trong lần cập nhật docs này. Một số module controller import helper từ
scripts/, vì vậy clone GitHub một mình chưa đủ để chạy workflow train/eval
controller hoặc bộ regression nội bộ.

[Requirements](../requirements.txt) và
[vLLM lock lịch sử](../requirements-vllm.lock.txt) không bảo đảm cài được trên
mọi nền tảng. Official MedQA exports, model cache/tokenizer, configs đóng băng
và runtime của từng job phải được chuẩn bị theo revision/manifest riêng.
Xem [module map](../controller/README.md) và [config inventory](../configs/README.md).

Chỉ trên server có bộ scripts/tests riêng, chạy từ repo root trong Conda latentmas:

~~~bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests
git diff --check
~~~

Lượt cleanup 27/10: 307 test, 306 pass, 1 historical skip; 9 test chỉ kiểm runner
đã bỏ được gỡ; 5 file test nhỏ được gộp và 307 method giữ nguyên nội dung.
Đây là CPU regression, không phải Qwen GPU parity hay kết quả training mới.

Nguồn, dataset/profile/checkpoint/log và snapshot của job cũ được giữ nguyên.
Source snapshot bất biến: thay working tree không sửa hash để resume job cũ.
Git index và các thay đổi ngoài phạm vi phải được review trước khi stage;
không suy trạng thái index từ ghi chú lịch sử trong tài liệu.

## 9. Quy ước tài liệu từ đây

Chỉ dùng file này làm docs chính. Không tự tạo lại báo cáo pilot, migration,
proposal, cleanup diary hoặc bảng kết quả của các hướng đã loại.
Không quay lại train/phân tích các nhánh bỏ chỉ vì còn checkpoint hoặc code.

Các lần train/check/fix code thông thường ghi trạng thái vào job logs/manifests,
không tự sửa docs. Chỉ cập nhật tài liệu hoặc khôi phục phần đã bỏ khi người dùng
yêu cầu rõ. Không lấy các ghi chú lịch sử làm mặc định ghi đè hướng hiện tại.

Bản sao docs trước khi gộp chỉ phục vụ phục hồi khi được yêu cầu:
 /data/nghialt/.remote-workspace/jobs/latentmas-docs-consolidate-20261021-v1/docs-before.tar.gz

## 10. Trạng thái job tại mốc cập nhật

| Job | Trạng thái đã kiểm | Kết quả được phép kết luận |
|---|---|---|
| KVComm ba nhánh test500 | Hoàn tất, đủ1500 B completions | Bảng §5,1 |
| A K10→selective KV→B K0/K10, multi-GPU v3 | Chờ VRAM, không có GPU worker; giữ 180/500 cặp từ v1 | Chưa có bảng accuracy toàn500; không dùng partial để chọn cấu hình |
| Shortcut A K2 distillation | waiting_native_profile; 0 update | Chưa train, chưa test; chưa có accuracy |
| Native adaptive campaign7K,seed42 | cancelled_by_user ngày27/10 | Chỉ 6/128 câu diagnostic train đã commit; 0/64 validation; không có profile đầy đủ |

Selective v3 chia phần còn lại thành [180,340) và[340,500), tối đa 2 GPU.
Mỗi worker chạy cả hai nhánh B0/B10 trên cùng cohort, batch32; OOM mới thử16;
cần ít nhất 24576MiB free/GPU. 180 cặp cũ dùng batch4 được ghi provenance riêng.
Gate thực trên Qwen phải chạy trước mỗi worker; test CPU không thay gate GPU.
Không cộng thời gian GPU song song rồi gọi là elapsed wall time.

Campaign native bốn stage trong plan trước đây đã bị người dùng hủy do chậm.
Không tự khởi động lại hoặc coi mô tả được-authorize cũ là queue còn hoạt động.
Các stage paired action values, B probe và A adapter của campaign chưa chạy.
K0 trong thiết kế đó vẫn giữ A prompt KV, không phải B-only.

Shortcut K2 vẫn bị chặn bởi native profile đã hủy; watcher còn chờ không có nghĩa
training đang chạy. Việc tận dụng dữ liệu hybrid cũ hoặc đổi sang gold-only mới
được thảo luận, chưa đổi objective/dependency. Cần quyết định riêng trước khi
chuyển protocol; không dùng correctness hybrid làm reward native mặc nhiên.

Các job directories dưới /data/nghialt/.remote-workspace/jobs/:
- latentmas-selective-B-latent-multigpu-20261027-v3
- latentmas-shortcut-K2-distill-20261027-v1
- latentmas-native-adaptive-after-kvcomm-20261027-v2 (đã hủy)

Đối chiếu status.json, complete markers, source/input hashes và exit code để
cập nhật trạng thái. Bảng này là snapshot 27/10, không phải live dashboard.
Lượt sửa README không đổi queue, chạy thêm model hoặc tinh chỉnh protocol.
