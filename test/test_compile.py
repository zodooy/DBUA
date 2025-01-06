import torch
import time
# torch._dynamo.config.suppress_errors = True

def time_evaluation(origin, compiled, input, exec_func=None, exp_name: str = '', warmup_time: int = 5) -> None:
    torch.cuda.synchronize()
    s_t = time.time()
    exec_func(origin, input) if exec_func else origin(input)
    torch.cuda.synchronize()
    start_t1 = time.time() - s_t
    

    torch.cuda.synchronize()
    s_t = time.time()
    exec_func(compiled, input) if exec_func else compiled(input)
    torch.cuda.synchronize()
    start_t2 = time.time() - s_t
    print(f"Normal firstly used time:{start_t1}s")
    print(f"Compiled firstly used time:{start_t2}s")

    assert warmup_time >= 1
    for _ in range(warmup_time - 1):
        exec_func(compiled, input) if exec_func else compiled(input)

    t_1_total, t_2_total = 0., 0.
    for i in range(10):
        torch.cuda.synchronize()
        s_t = time.time()
        exec_func(origin, input) if exec_func else origin(input)
        torch.cuda.synchronize()
        t_1 = time.time() - s_t
        t_1_total += t_1

        torch.cuda.synchronize()
        s_t = time.time()
        exec_func(compiled, input) if exec_func else compiled(input)
        torch.cuda.synchronize()
        t_2 = time.time() - s_t
        t_2_total += t_2

        print(f"{i}:\n\tNormal used time:{t_1}s, \n\t"
              f"Compiled used time:{t_2}s")

    print(f"{exp_name}在编译前的首次运行时间为:{start_t1}秒")
    print(f"{exp_name}在编译后的首次运行时间为:{start_t2}秒")
    print(f"{exp_name}在后续运行过程中的加速比为:{t_1_total / t_2_total:.2f}")

# 一个简单的函数
def simple_fn(x):
    for _ in range(20):
        y = torch.sin(x).cuda()
        x = x + y
    return x


compiled_fn = torch.compile(simple_fn, backend="cudagraphs")

input_tensor = torch.randn(10000).to(device="cuda:0")

# 测试
time_evaluation(simple_fn, compiled_fn, input_tensor, None, '简单函数')