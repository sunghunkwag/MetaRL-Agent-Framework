class MetaWorldDistribution(TaskDistribution):
    """Base class for MetaWorld distributions."""

    def __init__(self, benchmark_name: str):
        if metaworld is None:
            raise ImportError("MetaWorld is not installed. Please install it with: pip install metaworld")

        if benchmark_name == 'ml10':
            self.benchmark = metaworld.ML10()
        elif benchmark_name == 'ml45':
            self.benchmark = metaworld.ML45()
        else:
            raise ValueError(f"Unknown MetaWorld benchmark: {benchmark_name}")

        # Only unique env_name per split (ML10: 10 train, 5 test)
        train_taskmap = {}
        for task in self.benchmark.train_tasks:
            env_cls = self.benchmark.train_classes[task.env_name]
            if task.env_name not in train_taskmap:
                train_taskmap[task.env_name] = (env_cls, task, 'train')
        test_taskmap = {}
        for task in self.benchmark.test_tasks:
            env_cls = self.benchmark.test_classes[task.env_name]
            if task.env_name not in test_taskmap:
                test_taskmap[task.env_name] = (env_cls, task, 'test')

        self._all_tasks = list(train_taskmap.values()) + list(test_taskmap.values())
        super().__init__(f'metaworld-{benchmark_name}', self._all_tasks)

    def create_task(self, task_id: int):
        env_cls, task, split = self.task_params[task_id]
        env = env_cls()
        env.set_task(task)
        return env

    def get_train_test_split(self) -> Tuple[List[int], List[int]]:
        train_ids = [i for i, t in enumerate(self.task_params) if t[2] == 'train']
        test_ids = [i for i, t in enumerate(self.task_params) if t[2] == 'test']
        return train_ids, test_ids
