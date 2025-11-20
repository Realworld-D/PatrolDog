class FilterUtil:
    def __init__(self, timeRange):
        self.timeRange = timeRange
        self.pos = 0
        self.content = [0 for _ in range(timeRange)]
    
    def putData(self, x):
        self.content[self.pos] = x
        self.pos = (self.pos + 1) % self.timeRange
    
    def query(self):
        return max(self.content, key=self.content.count)
    
if __name__ == "__main__":
    filterUtil = FilterUtil(10)
    filterUtil.putData(1)
    print(filterUtil.content)
    print(filterUtil.query())
    for _ in range(5):
        filterUtil.putData(1)
    print(filterUtil.content)
    print(filterUtil.query())
    for i in range(5):
        filterUtil.putData(i)
    print(filterUtil.content)
        