from enum import IntEnum


class Actions(IntEnum):
    Defence1 = 0
    Defence2 = 1
    Defence3 = 2
    Defence4 = 3

    # get the enum name without the class
    def __str__(self):
        return self.name
