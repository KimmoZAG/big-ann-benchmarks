# MutilModel Retrieval On Dataflow

## 📖 Chapter 1 : Run Baseline

### 1.0 Complete the setup of the Python environment

``` 
git clone https://github.com/harsha-simhadri/big-ann-benchmarks.git
cd big-ann-benchmarks
conda create -n bann python=3.10
pip install -r requirements_py3.10.txt
```

PS:这里可能会报matplotlib的错，txt里面删除matplotlib的版本号即可

---

### 1.1 Build the docker container baseline

此举目的为了构建基于 **[Developed algorithms name]** 的镜像以供测试

- The baselines were run on an Azure Standard D8lds v5 (8 vcpus, 16 GiB memory) machine.
- 虽然 Docker 支持分配 vCPUs 大于物理核数，但最好还是在8核及以上的 CPU 上运行

``` python
python install.py --neurips23track streaming --algorithm diskann
python install.py --neurips23track streaming --algorithm pyanns
python install.py --neurips23track streaming --algorithm [Developed algorithms name]
```

---

### 1.2 Test the benchmark using the algorithm's definition file

在data/random10000下存有data_10000_768，queries_1000_768_gt，queries_1000_768_query文件

#### 1.2.1 Streaming scenario testing based on ANN

为了测试ANN在流数据上的性能，更改**queries_1000_768_gt**为**queries_1000_768**，运行：

```
python run.py --neurips23track streaming --algorithm diskann --dataset random-xs --runbook_path neurips23/runbooks/simple_runbook.yaml
python run.py --neurips23track streaming --algorithm pyanns --dataset random-xs --runbook_path neurips23/runbooks/simple_runbook.yaml
```

在results/neurips23/streaming下得到相应的hdf5文件，该文件存储了每次search的结果[ANN算法搜索到的'每个query的'K个最近邻的'序列号<===>(n, querynum, k)]

---

#### 1.2.2 Rename

将**queries_1000_768**改回**queries_1000_768_gt**

---

#### 1.2.3 Streaming scenario testing based on KNN

为了得到groundtruth以供测试recall，更改**queries_1000_768_query**为**queries_1000_768**

首先要安装一个专用于streaming的评估工具

``` 
git clone https://github.com/microsoft/DiskANN.git
cd DiskANN
mkdir build
cd build
cmake ..
make
```

开始进行计算GroundTruth

``` python
python benchmark/streaming/compute_gt.py --dataset random-xs --runbook neurips23/runbooks/simple_runbook.yaml --gt_cmdline_tool DiskANN/build/apps/utils/compute_groundtruth
```

---

### 1.3 计算 Recall

```
sudo chmod 777 -R results/
python data_export.py --out res.csv
```

---

## 📖 Chapter 2 : More Detail of Baseline

### 2.1 About Docker

利用neurips23/streaming/[Developed algorithms name]下的Dockerfile进行安装(big-ann的源码可能已经在docker中)
- 安装依赖
- 下载算法的源代码
- 设置工作目录
- 安装 Python 环境和构建工具
- 构建并安装算法的 Python 包

---

### 2.2 About Runbook

Runbook位于neurips23/runbooks/下，有多个runbook可供选择，每个runbook中有表明可用的数据集

以simple_runbook.yaml为例
``` python
random-xs:                      # 可用的数据集
  max_pts: 10000                # 最大点数(上限)
  1:                            # 动作编号
    operation: "insert"         # 动作名称
    start: 0                    # 开始点编号
    end: 10000                  # 结束点编号
  2:
    operation: "search"
  3:
    operation: "delete"
    start: 0
    end: 5000
  4:
    operation: "search"
  5:
    operation: "insert"
    start: 0
    end: 5000
  6:
    operation: "search"
  gt_url: "https://comp21storage.z5.web.core.windows.net/comp23/str_gt/random10000/10000/simple_runbook.yaml" # 用于下载真值
...
```

---

### 2.3 About Run.py 

#### 2.3.1 首先是通过run_docker进行容器实例化

``` python
def run_docker(definition, dataset, count, runs, timeout, rebuild,
               cpu_limit, mem_limit=None,
               t3=None, power_capture=None,
               upload_index=False, download_index=False,
               blob_prefix="", sas_string="", private_query=False,
               neurips23track='none', runbook_path='neurips23/streaming/simple_runbook.yaml')
```

- dataset: random-xs
- count: 10
- runs: 5
- timeout: 43200
- rebuild: False
- cpu_limit: 0-11
- mem_limit: 26611702528
- t3: False
- power_capture: 
- upload_index: False
- download_index: False
- blob_prefix: None
- sas_string: None
- private_query: False
- neurips23track: streaming
- runbook_path: neurips23/runbooks/simple_runbook.yaml

具体运行时的传参以及其他设置都在以下代码设定：

``` python
container = client.containers.run(
    definition.docker_tag,
    cmd,
    volumes={
        os.path.abspath('benchmark'):
            {'bind': '/home/app/benchmark', 'mode': 'ro'},
        os.path.abspath('data'):
            {'bind': '/home/app/data', 'mode': 'rw'},
        os.path.abspath('results'):
            {'bind': '/home/app/results', 'mode': 'rw'},
        os.path.abspath('neurips23'):
            {'bind': '/home/app/neurips23', 'mode': 'ro'},
    },
    cpuset_cpus=cpu_limit,
    mem_limit=mem_limit,
    detach=True)
```

cmd的内容，具体传入的程序是run_algorithm.py这个程序
- ['--dataset', 'random-xs', '--algorithm', 'pyanns', '--module', 'neurips23.streaming.pyanns.pyanns', '--constructor', 'Pyanns', '--runs', '5', '--count', '10', '--neurips23track', 'streaming', '--runbook_path', 'neurips23/runbooks/simple_runbook.yaml', '["euclidean", {"R": 32, "L": 50, "insert_threads": 16, "consolidate_threads": 16}]', '[{"Ls": 50, "T": 8}]']


#### 2.3.2 其次是执行run_algorithm.py程序

具体获取算法的config.yaml的代码在benchmark/main.py下：

``` python
definitions = get_all_definitions(
            neurips23.common.track_path(args.neurips23track), 
            dimension, args.dataset, distance, args.count)
```

随后会根据 **算法-数据-Runbook** 进行streaming测试

---

### 2.4 相关指标说明

不论是run.py还是benchmark/streaming/compute_gt.py，都没有进行Recall的计算

run.py结束后，得到的hdf5文件：

```
algo: pyanns
build_time: 0.0064868927001953125               
count: 10
dataset: random-xs
distance: euclidean
index_size: 11612.0
name: pyanns(('R32_L50', {'Ls': 50, 'T': 8}))
num_searches: 3
private_queries: False
run_count: 1
search_times: []
step_0: 2
step_1: 4
step_2: 6
type: knn
Datasets and Groups:
Reading group: /
Dataset neighbors_step2:
[8101 8514 7743  523 1358 5143 5327 9704 5193 7686]
...
Dataset neighbors_step4:
[8101 8514 7743 5143 5327 9704 5193 7686 7488 6458]
...
```

benchmark/streaming/compute_gt.py结束后，得到的文件：

```
[[6654. 7213. 6888. ... 4614.  267. 3560.]
...
 [6654. 7213. 6888. ... 3560. 5956. 4614.]]
[[22557.51367188 22759.91210938 22793.02539062 ... 24223.93359375
  24226.04492188 24230.625     ]
...
 [22570.55273438 22774.61914062 22806.61914062 ... 24225.58984375
  24230.97851562 24235.41210938]]
```

是两个(1000, 100)的数据，第一个knn下的k个最近邻的id,第二个是距离query的相应距离

data_export.py最终会完成一些性能指标的计算，目前Streaming好像只有Recall


| Algorithm | Parameters                                        | Dataset                         | Count | DistComps | Build | IndexSize | Mean SSD IOs | Mean Latency | Track   | Recall/AP |
|-----------|---------------------------------------------------|---------------------------------|-------|-----------|-------|-----------|--------------|--------------|---------|-----------|
| diskann   | diskann(('R64_L50', {'Ls': 100, 'T': 16}))       | msturing-10M-clustered(delete_runbook.yaml) | 10    | 0.0       | 0.8402106761932373 | 2847936.0   | 0.0          | 0.0          | streaming | 0.8288063636363635 |


``` python
all_metrics = {
    "k-nn": {
        "description": "Recall",
        "function": lambda true_nn, run_nn, metrics, run_attrs: knn(true_nn, run_nn, run_attrs["count"], metrics).attrs['mean'],  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "distcomps": {
        "description": "Distance computations",
        "function": lambda true_nn, run_nn,  metrics, run_attrs: dist_computations(len(true_nn[0]), run_attrs), # noqa
        "worst": float("inf")
    },
    "build": {
        "description": "Build time (s)",
        "function": lambda true_nn, run_nn, metrics, run_attrs: build_time(run_attrs), # noqa
        "worst": float("inf")
    },
    "indexsize": {
        "description": "Index size (kB)",
        "function": lambda true_nn, run_nn, metrics, run_attrs: index_size(run_attrs),  # noqa
        "worst": float("inf")
    },
    "mean_ssd_ios": {
        "description": "Average SSD I/Os per query",
        "function": lambda true_nn, run_nn, metrics, run_attrs: mean_ssd_ios(run_attrs),  
        "worst": float("inf")
    },
    "mean_latency": {
        "description": "Mean latency across queries",
        "function": lambda true_nn, run_nn, metrics, run_attrs: mean_latency(run_attrs),  
        "worst": float("inf")
    },
}
```

---

## 📖 Chapter 3 : Cirr Baseline

### 3.1 Prepare Data

``` python
python create_dataset.py --dataset cirr
```

python run.py --neurips23track streaming --algorithm diskann --dataset cirr --runbook_path neurips23/runbooks/simple_runbook.yaml
python benchmark/streaming/compute_gt.py --dataset cirr --runbook neurips23/runbooks/simple_runbook.yaml --gt_cmdline_tool DiskANN/build/apps/utils/compute_groundtruth