class DecisionLog:
    def __init__(self):
        self.entries = []

    def listener(self, event, payload):
        if event == "optimization.decision":
            self.entries.append(payload)
