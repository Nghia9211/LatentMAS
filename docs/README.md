# RL Controller — thiết kế và phân tích hiện tại

Tài liệu chính của nhánh adaptive latent handoff. Chốt nội dung ngày 21/10/2026.
Các báo cáo pilot, migration, proposal và bảng kết quả đã loại không còn là
hướng dẫn làm việc. Không tự khôi phục hoặc tiếp tục các nhánh đó.

## 1. Bài toán và phạm vi

Hai agent Qwen3-4B được giữ cố định: A tạo latent trajectory, B nhận prefix để
sinh đáp án. Controller học chọn thời điểm chuyển giao theo từng câu, nhằm
tăng strict correctness của B. Chưa có kết luận adaptive tốt hơn fixed budget.

- A tạo latent input e_t từ h_(t-1) qua realignment rồi forward để nhận h_t.
- h_0 là last-layer hidden tại token thật cuối của prompt sau prefill.
- Tại checkpoint k, current hidden là h_k; previous checkpoint là h_(k-10).
- Prefix chuyển cho B chứa latent inputs e_1…e_k; không đồng nhất e_t với h_t.
- Chỉ cập nhật controller; không backprop qua A/B hoặc fine-tune Qwen.

## 2. State hiện tại: ba nhóm

Schema: hk_cosine10_k_train_zscore_v1. Tổng cộng 2.562 chiều.

| Nhóm | Giá trị | Xử lý trong controller |
|---|---|---|
| Hidden của A | h_k, 2.560 chiều | LayerNorm riêng theo observation, epsilon 1e-5 |
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
| PPO | clip=0.2, value weight=0.5, entropy weight=0.01 |
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

## 4. Training đang ưu tiên: fresh rollout PPO

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
[entrypoint](../scripts/train_cosine_state_ppo.py),
[config gần nhất](../configs/controller_ppo_cosine_state64_offline_v2.yaml).

## 5. Mốc kết quả và trạng thái cần nhớ

| Mốc giữ lại | Bằng chứng / ý nghĩa |
|---|---|
| Fixed K10 trên cùng test500 đã chạy | 348/500 = 69,6%; meanK=10. Đây là mốc tham chiếu đã quan sát, không phải fixed budget được chọn mới bằng test |
| Controller L2 đã kiểm tra | Chưa vượt fixed K10; giữ kết luận để định hướng, bỏ các bảng pilot/ablation yếu khỏi tài liệu chính |
| Cosine64 v2 | Dừng ở khởi tạo vLLM, exit1, 0 committed PPO iterations; chưa có accuracy của controller cosine |

Ở lần lỗi cosine v2, vLLM ghi free memory 9,65 GiB, thấp hơn mức yêu cầu
14,25 GiB. Lỗi khởi tạo không phải kết quả chất lượng của state cosine.
Sau các thay đổi source, cần prepare config/output/job version mới trước khi
train lại; không sửa manifest/hash cũ để resume cho qua kiểm tra.

Bộ 500 câu đã được dùng trong các lần trước; không gọi đó là held-out chưa từng
xem. Chênh lệch một lần decode chưa chứng minh cải thiện có thể lặp lại.

Nguồn kiểm tra giữ ngoài docs:
- /data/nghialt/.remote-workspace/jobs/latentmas-test500-simple-state-best40-20261021-v1/verified_results.json
- /data/nghialt/.remote-workspace/jobs/latentmas-test500-simple-state-last200-20261021-v1/verified_results.json
- /data/nghialt/.remote-workspace/jobs/latentmas-ppo-cosine-state64-20261021-v2/state.json

Không xóa hoặc sửa raw results để biến kết luận chưa tốt thành cải thiện.

## 6. Phân tích còn cần theo dõi

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

Bước nghiên cứu tiếp theo là kiểm tra cosine với cùng reward/kiến trúc và live
rollout, sau khi giải quyết điều kiện khởi tạo GPU. Không tự mở thêm ablation,
đổi state/reward hoặc kéo dài training.

## 7. Evaluation và logging

Chọn checkpoint bằng validation accuracy; hòa thì lower meanK, rồi iteration
sớm hơn. Khóa checkpoint trước test. Không chọn lại theo test accuracy.

Giữ Qwen3-4B BF16, pinned revision theo manifest, prompt sequential/realignment,
B temperature0.6/top_p0.95/top_k0/cap4096/batch16 và seed theo protocol.
Manual think=false không đồng nghĩa chat-template enable_thinking=false.
Giữ padding masks, logical positions và ragged latent prefixes khi handoff.

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

## 8. Vận hành, kiểm tra và chuẩn bị GitHub

Workspace /data/nghialt; repo /data/nghialt/LatentMAS; SSH alias brev-h100.
Runtime Conda latentmas, prefix /data/nvidia/conda_envs/latentmas.
Private tmux socket: /data/nghialt/.remote-workspace/tmux.sock.
Không thao tác process của người khác; kiểm identity/start time/task provenance.

[Requirements server](../requirements-server.txt) ghi runtime H100; không đồng
nhất với lock GB10 hoặc bảo đảm cài được trên mọi nền tảng.
[Tracking dependencies](../requirements-tracking.txt) là phần W&B.
Official MedQA exports và model cache/tokenizer phải chuẩn bị riêng theo revision
và manifest; không có trong source-only ZIP. Giữ license/attribution upstream.

[Script map](../scripts/README.md), [module map](../controller/README.md),
[config inventory](../configs/README.md).

~~~bash
# Chạy từ repo root, trong runtime đã chọn.
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
git diff --check
~~~

Lượt cleanup code gần nhất: 252 tests, 251 pass và 1 historical skip;
real L2/cosine checkpoints giữ nguyên state dict, observations và forward outputs.
Đây là CPU/checkpoint verification, không phải kết quả training cosine mới.

Một số tests cần official exports, cached tokenizer hoặc historical artifacts.
Thiếu dữ liệu không được báo là parity đã pass. Dataset/profile/checkpoint/log
ở các thư mục generated vẫn được giữ và ignore khỏi Git.

Worktree hiện có staged changes từ trước. Phải review cả working tree và index
trước commit; cleanup này không tự stage/commit/push. Source snapshots và
manifest của run cũ bất biến; thay code cần source snapshot/version mới.

## 9. Quy ước tài liệu từ đây

Chỉ dùng file này làm docs chính. Không tự tạo lại báo cáo pilot, migration,
proposal, cleanup diary hoặc bảng kết quả của các hướng đã loại.
Không quay lại train/phân tích các nhánh bỏ chỉ vì còn checkpoint hoặc code.

Các lần train/check/fix code thông thường ghi trạng thái vào job logs/manifests,
không tự sửa docs. Chỉ cập nhật tài liệu hoặc khôi phục phần đã bỏ khi người dùng
yêu cầu rõ. Không lấy các ghi chú lịch sử làm mặc định ghi đè hướng hiện tại.

Bản sao docs trước khi gộp chỉ phục vụ phục hồi khi được yêu cầu:
 /data/nghialt/.remote-workspace/jobs/latentmas-docs-consolidate-20261021-v1/docs-before.tar.gz
