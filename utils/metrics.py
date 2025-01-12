#
from collections import defaultdict

class Metrics:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_metrics(self):
        return {k: sum(v)/len(v) for k, v in self.metrics.items()}
    
    def get_all_metrics(self):
        return self.metrics
    

#cloner174