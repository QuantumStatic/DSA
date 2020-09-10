from myFunctionsG import nearMatching
from statistics import mean
from random import shuffle
from copy import deepcopy


# For Heaps
maxHeap, minHeap = True, False

class Heap:
    __slots__ = ["keys", "k", "heapType"]
    def __init__(self, *values, heapType=None, kValue=2):
        self.heapType, self.k, self.keys = heapType, kValue, list()
        self.addElements(values)

    def __str__(self):
        return str(self.keys)

    def addElements(self, *values):
        for value in Heap.breaker(values):
            self.keys.append(value)
        self.buildHeap()

    @staticmethod
    def breaker(toBreak):
        for value in toBreak:
            if isinstance(value, int) or isinstance(value, float):
                yield value
            elif isinstance(value, set) or isinstance(value, list) or isinstance(value, tuple):
                yield from Heap.breaker(value)
            else:
                raise TypeError(f"{value} of type {type(value).__name__} is not supported.")

    def kids(self, index):
        for n in range(1,self.k+1):
            yield (self.k*index+n)

    def Heapify(self, index):
        if self.heapType is maxHeap:
            largest = index
            for kid in self.kids(index):
                if kid >= len(self.keys):
                    break
                if self.keys[kid] > self.keys[largest]:
                    largest = kid
            if largest != index:
                self.keys[largest], self.keys[index] = self.keys[index], self.keys[largest]
                self.Heapify(largest)
        elif self.heapType is minHeap:
            smallest = index
            for kid in self.kids(index):
                if kid >= len(self.keys):
                    break
                if self.keys[kid] < self.keys[smallest]:
                    smallest = kid
            if smallest != index:
                self.keys[smallest], self.keys[index] = self.keys[index], self.keys[smallest]
                self.Heapify(smallest)

    def buildHeap(self):
        if self.heapType is not None:
            for index in range(len(self.keys)//self.k, -1,-1):
                self.Heapify(index)

    def extract(self):
        if len(self.keys) == 0:
            return None
        temp = self.keys[0]
        del self.keys[0]
        self.Heapify(0)
        return (temp)

    def changeHeapType(self):
        if self.heapType is not None:
            self.heapType = not self.heapType
        else: 
            raise Exception("Can't change Heap when no heap type is assingned. Try using setHeapType function instead")

    def setHeapType(self,HeapType):
        if isinstance(HeapType, bool):
            self.heapType = HeapType
        else:
            raise TypeError(f"Entered type {type(HeapType)} is not supported. try writing maxHeap or minHeap")

    def sort(self):
        sortedkeys = list()
        if self.heapType is minHeap:
            while (temp := self.extract()) is not None:
                sortedkeys.append(temp)
        elif self.heapType is maxHeap:
            while (temp := self.extract()) is not None:
                sortedkeys.insert(0, temp)
        else:
            raise TypeError("can't sort a heap when its type is not specified, try using list sort method instead")
        self.keys = sortedkeys
        del sortedkeys

# Declare a Heap as:
#   foo = Heap(list, heapType=MaxHeap) or foo = Heap(tuple, heapType=MinHeap, kValue=3) or foo = Heap(int, int, float, int, float...,heapType=MinHeap, kValue=67)

class PriorityQueue:

    class item:
        __slots__ = ["priority", "task", 'id']
        def __init__(self, priority:int, task, idnum=None):
            self.priority, self.task, self.id = priority, task, idnum
        
        def __str__(self):
            return str({"Task":self.task, "Priority":self.priority, "ID":self.id})

        def __call__(self):
            return self

        def copy(self):
            return deepcopy(self)

    __slots__ = ['items', 'currTask', 'totalItems']
    def __init__(self, *args, priorityList=None, taskList=None):
        self.items, self.currTask, self.totalItems = list(), None, int()
        if priorityList is not None and taskList is not None:
            data = zip(priorityList, taskList)
            self.addItems(data)
        if any(args):
            self.addItems(args)

    def __str__(self):
        copyInstance, final = self.copy(), str()
        while any(copyInstance.items):
            final += str(copyInstance.getMostImportanTask().task) + '\n'
            copyInstance.markDone()
            # for x in copyInstance.items:
                # print(x)
        return final

    def addItems(self, Items):
        if isinstance(Items, dict):
            for x in Items.keys():
                self.items.append(self.item(priority=x, task=Items[x], idnum=self.totalItems))
                self.totalItems += 1
        elif isinstance(Items, self.item):
            self.items.append(Items)
            self.totalItems += 1
        elif (isinstance(Items, tuple) or isinstance(Items, list)) and len(Items) == 2:
            self.items.append(self.item(priority=Items[0], task=Items[1], idnum=self.totalItems))
            self.totalItems += 1
        elif isinstance(Items, list) or isinstance(Items, tuple) or isinstance(Items, set):
            for item in Items:
                self.addItems(item)
        else:
            raise TypeError(f"item {Items} of type {type(Items).__name__} is not supported")
        self.buildQueue()

    def createItem(self, priority:int, task):
        self.addItems(self.item(priority, task))

    def kids(self, index):
        for n in range(1,2+1):
            yield(2*index+n)

    def minHeapify(self, index):
        smallest = index
        for kid in self.kids(index):
            if kid >= self.totalItems:
                break
            if self.items[kid].priority < self.items[smallest].priority:
                smallest = kid
        if smallest != index:
            self.items[smallest].id, self.items[index].id = self.items[index].id, self.items[smallest].id
            self.items[smallest], self.items[index] = self.items[index], self.items[smallest]
            self.minHeapify(smallest) 

    def buildQueue(self):
        for index in range(self.totalItems//2, -1,-1):
            self.minHeapify(index)

    def getTaskbyPriority(self, priority:int):
        for x in self.items:
            if x.priority == priority:
                self.currTask = x
                return x()

    def getTaskbyId(self, Idnum:int):
        self.currTask = self.items[Idnum]
        return self.items[Idnum]()

    def getMostImportanTask(self):
        self.currTask = self.items[0]
        return self.items[0]()

    def markDone(self, toMarkDone=None):
        if toMarkDone is not None and toMarkDone != self.currTask.id and isinstance(toMarkDone, int) and toMarkDone <= self.totalItems:
            for x in range(toMarkDone+1, self.totalItems):
                self.items[x].id -= 1
            del self.items[toMarkDone]
        elif isinstance(toMarkDone, self.item):
            for x in range(toMarkDone.id+1, self.totalItems):
                self.items[x].id -= 1
            del self.items[toMarkDone.id]
        elif toMarkDone is None:
            if self.currTask is None:
                self.currTask = self.getMostImportanTask()
            for x in range(self.currTask.id+1, self.totalItems):
                self.items[x].id -= 1
            del self.items[self.currTask.id]
        else:
            raise TypeError(f"passed argument {toMarkDone} of type {type(toMarkDone).__name__} is not supported")
        self.totalItems -= 1
        self.buildQueue()

    def copy(self):
        return deepcopy(self)

# Declare a PriorityQueue as:
#   1 IS CONSIDERED HIGHEST PRIORITY (1>2>3>4...)
#   foo = PriorityQueue(dict1, dict2, dict3...) where the dictionary is of the format {priority(must be a number):task}
#   foo = PriorityQueue(priorityList=[], taskList=[]) where priorityList is a list(or tuple) priorities(numbers) & where taskList is a list(or tuple) of <any datatype>



# For RedBlack Trees
red, black = False, True 

class Node:
    __slots__ = ['parent', 'right', 'left', 'value', 'repititions', 'colour']
    def __init__(self, parent=None, rightChild=None, leftChild=None, value=int(), repititions=int(), colour=red):
        self.parent, self.right, self.left, self.value, self.repititions, self.colour = parent, rightChild, leftChild, value, repititions, colour

    def __str__(self):
        return str({'value':self.value, 'parent':self.parent.value if self.parent is not None else None, 'right':self.right.value if self.right is not None else None, 'left':self.left.value if self.left is not None else None, 'colour':'Black' if self.colour == black else 'Red', 'repititions': self.repititions})

    def __repr__(self):
        return NotImplemented

    def __call__(self):
        return (self)

    def __eq__(self, other):
        if self is not None and other is not None:
            if self.value == other.value:
                if self.right == other.right:
                    if self.left == other.left:
                        if self.colour == other.colour:
                            if self.repititions == other.repititions:
                                return True
        return False

    def copy(self):
        return deepcopy(self)


class BinarySearchTree:

    class leaf(Node):
        def __init__(self, parent=None, rightChild=None, leftChild=None, value=int(), repititions=int()):
            super().__init__(parent=parent, rightChild=rightChild, leftChild=leftChild, value=value, repititions=repititions)

        def __str__(self):
            return str({'value':self.value, 'parent':self.parent.value if self.parent is not None else None, 'right':self.right.value if self.right is not None else None, 'left':self.left.value if self.left is not None else None, 'repititions': self.repititions})

    sentinel = None

    def __init__(self, *args):
        elements = BinarySearchTree.breaker(args)
        index = nearMatching(elements,mean(elements))
        self.root = self.leaf(value=elements[index], parent=BinarySearchTree.sentinel)
        del elements[index]
        shuffle(elements)
        self.insert(elements)

    def __str__(self):
        returnString = '|< '
        def inordertreewalk(itr):
            nonlocal returnString
            if itr.left != eval(type(self).__name__).sentinel:
                inordertreewalk(itr.left)
            returnString += f"{itr.value} "
            if itr.right != eval(type(self).__name__).sentinel:
                inordertreewalk(itr.right)
        inordertreewalk(self.root)
        returnString += '>|'
        return returnString

    def __repr__(self):
        return NotImplemented

    @staticmethod
    def breaker(toBreak):
        final = list()
        for x in toBreak:
            if isinstance(x,int):
                final.append(x)
            elif isinstance(x, list) or isinstance(x,tuple) or isinstance(x,set):
                final.extend(BinarySearchTree.breaker(x))
            else:
                raise TypeError(f"{x} of type {type(x)} isn't supported yet")
        return final

    def insert(self,*toAdd):
        newElements = BinarySearchTree.breaker(toAdd)
        for num in newElements:
            iterator = self.root
            while True:
                if num == iterator.value:
                    iterator.repititions += 1
                    break
                elif num > iterator.value:
                    if iterator.right is not BinarySearchTree.sentinel:
                        iterator = iterator.right
                    else:
                        iterator.right = self.leaf(parent=iterator, value=num)
                        break
                else:
                    if iterator.left is not BinarySearchTree.sentinel:
                        iterator = iterator.left
                    else:
                        iterator.left = self.leaf(parent=iterator, value=num)
                        break

    def find(self, value,returnNode=False):
        iterator = self.root
        while iterator != eval(type(self).__name__).sentinel:
            if value == iterator.value:
                if returnNode:
                    return iterator()
                return True
            if value > iterator.value:
                iterator = iterator.right 
            else:
                iterator = iterator.left 
        return False

    def iterGenerator(self, node):
        if node is None:
            return self.root
        if isinstance(node, self.leaf):
            return node
        elif isinstance(node, int):
            if (temp := self.find(node, returnNode=True)) is not False:
                return temp
            else:
                raise ValueError(f"value {node} doesn't exist in Tree")
        else:
            raise TypeError(f"object of type {type(node)} doesn't exist in tree")

    def max(self, tree=None, returnNode=False):
        iterator = self.iterGenerator(tree)
        while True:
            if iterator.right == eval(type(self).__name__).sentinel:
                if returnNode:
                    return iterator()
                return iterator.value
            iterator = iterator.right

    def min(self, tree=None, returnNode=False):
        iterator = self.iterGenerator(tree)
        while True:
            if iterator.left == eval(type(self).__name__).sentinel:
                if returnNode:
                    return iterator()
                return iterator.value
            iterator = iterator.left

    def next(self, value, returnNode=False):
        iterator = self.iterGenerator(value)
        if iterator.value == self.max():
            return None
        if iterator.right != eval(type(self).__name__).sentinel:
            return self.min(tree=iterator.right, returnNode=returnNode)
        while True:
            if iterator.parent.left is iterator:
                if returnNode:
                    return iterator.parent 
                return iterator.parent.value
            iterator = iterator.parent

    def prev(self, value, returnNode=False):
        iterator = self.iterGenerator(value)
        if iterator.value == self.min():
            return None
        if iterator.left != eval(type(self).__name__).sentinel:
            return self.max(tree=iterator.left, returnNode=returnNode)
        while True:
            if iterator.parent.right is iterator:
                if returnNode:
                    return iterator.parent
                return iterator.parent.value

    def transplant(self, depriciatedTree, newTree):
        node = self.iterGenerator(depriciatedTree)
        if not isinstance(newTree, self.leaf) and isinstance(newTree, int):
            newTree = self.leaf(value=newTree)
        elif newTree is None or isinstance(newTree, Node):
            pass
        elif not isinstance(newTree, int):
            raise TypeError(f"{type(newTree)} is not supported yet.")
        if node is self.root:
            self.root = newTree
        elif node.parent.right is node:
            node.parent.right = newTree
        else:
            node.parent.left = newTree
        try:
            newTree.parent = node.parent
        finally:
            return node

    def remove(self, toRemove):
        toDelete = self.iterGenerator(toRemove)
        if toDelete.repititions:
            temp = toDelete.copy()
            toDelete.repititions -=1
            return temp
        if toDelete.left is BinarySearchTree.sentinel:
            return self.transplant(toDelete, toDelete.right)
        elif toDelete.right is BinarySearchTree.sentinel:
            return self.transplant(toDelete, toDelete.left)
        else:
            replacement = self.next(toDelete)
            self.transplant(replacement, replacement.right)
            replacement.right, replacement.left = toDelete.right, toDelete.left
            return self.transplant(toDelete, replacement)

# Declare a BinarySearchTree as:
#   foo =  BinarySearchTree(int1, int2, int3...) or foo = BinarySearchTree(list) or BinarySearchTree(list1, list2, list3...) also works for tuples & sets

class RedBlackTree(BinarySearchTree):

    class leaf(Node):
        def __init__(self, parent=None, rightChild=None, leftChild=None, value=int(), repititions=int(), colour=red):
            super().__init__(parent=parent, rightChild=rightChild, leftChild=leftChild, value=value, repititions=repititions, colour=colour)

    sentinel = Node(parent=None, value=None, colour=black, rightChild=None, leftChild= None)

    def __init__(self, *args):
        elements = RedBlackTree.breaker(args)
        self.root = self.leaf(value=elements[0], parent=RedBlackTree.sentinel.copy(), rightChild=RedBlackTree.sentinel.copy(), leftChild=RedBlackTree.sentinel.copy(), colour=black)
        del elements[0]
        self.insert(elements)

    def rotate(self, parentLeaf, left=False, right=False):
        # Parent based rotation
        if left and right:
            raise Exception("Rotation direction unclear")
        if left: 
            if parentLeaf.right == RedBlackTree.sentinel:
                raise Exception("Type None rotations are not supported")
            rightChild, parentLeaf.right = parentLeaf.right, parentLeaf.right.left
            if rightChild.left != RedBlackTree.sentinel:
                rightChild.left.parent = parentLeaf
            rightChild.parent = parentLeaf.parent
            if parentLeaf.parent == RedBlackTree.sentinel:
                self.root = rightChild
            elif parentLeaf.parent.right is parentLeaf:
                parentLeaf.parent.right = rightChild
            else:
                parentLeaf.parent.left = rightChild
            rightChild.left, parentLeaf.parent = parentLeaf, rightChild
            return
        if right:
            if parentLeaf.left == RedBlackTree.sentinel:
                raise Exception("Type None rotations are not supported")
            leftChild, parentLeaf.left = parentLeaf.left, parentLeaf.left.right
            if leftChild.right != RedBlackTree.sentinel:
                leftChild.right.parent = parentLeaf
            leftChild.parent = parentLeaf.parent
            if parentLeaf.parent == RedBlackTree.sentinel:
                self.root=leftChild
            elif parentLeaf.parent.left is parentLeaf:
                parentLeaf.parent.left = leftChild
            else:
                parentLeaf.parent.right = leftChild
            leftChild.right, parentLeaf.parent = parentLeaf, leftChild
            return 
        raise Exception("Direction of rotation not specified")

    def insert(self,*toAdd):
        newElements = RedBlackTree.breaker(toAdd)
        for num in newElements:
            iterator = self.root
            while True:
                if num == iterator.value:
                    iterator.repititions += 1
                    break
                elif num > iterator.value:
                    if iterator.right != RedBlackTree.sentinel:
                        iterator = iterator.right
                    else:
                        iterator.right = self.leaf(parent=iterator, value=num, rightChild=RedBlackTree.sentinel.copy(), leftChild=RedBlackTree.sentinel.copy())
                        self.insertionFixup(iterator.right)
                        break
                else:
                    if iterator.left != RedBlackTree.sentinel:
                        iterator = iterator.left
                    else:
                        iterator.left = self.leaf(parent=iterator, value=num,rightChild=RedBlackTree.sentinel.copy(), leftChild=RedBlackTree.sentinel.copy())
                        self.insertionFixup(iterator.left)
                        break 

    def insertionFixup(self, AddedElement):
        # They are 3 cases in general,
        # Case 1: Inserted leaf's uncle is red then make the father and the uncle black and
        # make inserted node's grandfather red and shift index form inserted node to grandfather (the now trouble causing node)
        # Case 2: Inserted Leaf's uncle is black and it's parent is a right child and the inserted node itself is a left child(forming a triangle), 
        # perform a left rotation on this node to make it a left child (similarly if it's parent left child and the inserted node a right we perform a right rotation)
        # Case 3: inserted leaf's parent is a right child and inserted leaf is a right child
        # Basically the counterpart of Case 2, child oriented just like its parent. parent is right and so is its kid

        currLeaf = self.iterGenerator(AddedElement)
        while currLeaf.parent.colour is red:
            if currLeaf.parent is currLeaf.parent.parent.left:
                uncle_currLeaf = currLeaf.parent.parent.right
                if uncle_currLeaf is red:
                    uncle_currLeaf = currLeaf.parent = black 
                    currLeaf.parent.parent.colour, currLeaf = red, currLeaf.parent.parent             #case 1
                else:
                    if currLeaf.parent.right is currLeaf:
                        currLeaf=currLeaf.parent                                                      #case 2
                        self.rotate(currLeaf, left=True)
                    currLeaf.parent.colour, currLeaf.parent.parent.colour = black, red                #case 3
                    self.rotate(currLeaf.parent.parent, right=True)
            else:
                uncle_currLeaf = currLeaf.parent.parent.left
                if uncle_currLeaf is red:
                    uncle_currLeaf = currLeaf.parent = black
                    currLeaf.parent.parent.colour, currLeaf = red, currLeaf.parent.parent               # case 1
                else:
                    if currLeaf.parent.left is currLeaf:
                        currLeaf=currLeaf.parent                                                        # case 2
                        self.rotate(currLeaf, right=True)
                    currLeaf.parent.colour, currLeaf.parent.parent.colour = black, red                  # case 3
                    self.rotate(currLeaf.parent.parent, left=True)
        self.root.colour = black

    def remove(self, toRemove):
        toDelete = self.iterGenerator(toRemove)
        if toDelete.repititions:
            temp = toDelete.copy()
            toDelete.repititions -=1
            return temp
        suspected_discrepant_leaf, ogColour =  self.leaf(), toDelete.colour
        if toDelete.left == RedBlackTree.sentinel:
            suspected_discrepant_leaf = toDelete.right
            self.transplant(toDelete, toDelete.right)
        elif toDelete.right == RedBlackTree.sentinel:
            suspected_discrepant_leaf = toDelete.left 
            self.transplant(toDelete, toDelete.left)
        else:
            replacement = self.next(toDelete, returnNode=True)
            ogColour, suspected_discrepant_leaf = replacement.colour, replacement.right
            if replacement.parent is toDelete:
                suspected_discrepant_leaf.parent = replacement      #there might be a sentinel here
            else:
                self.transplant(replacement, replacement.right)
                replacement.right = toDelete.right 
                replacement.right.parent = replacement
            self.transplant(toDelete,replacement)
            replacement.left, replacement.colour = toDelete.left, toDelete.colour 
            replacement.left.parent = replacement
        if ogColour is black:
            self.deletionFixup(discrepantLeaf=suspected_discrepant_leaf)
        return toDelete

    def deletionFixup(self, discrepantLeaf):
        # They are 4 cases for deletion fixup
        # Since inserted node is red, we always look at the sibling of the inserted node 
        # Case 1: If sibling is red then make sibling black and make inserted leaf's parent red
        #         rotate the sibling such that the parent rotates towards the inserted node. 
        #         (ie. if inserted was a left child then the parent is left rotated)
        #         this changes inserted node's sibling. 
        # Case 2: Inserted Node's sibling's kids are both black 
        #         (irrespective sibling's colour, it might seem confusing but for now don't think about what we just did in Case 1)
        #         make sibling's colour red & colour inserted node's colour of the same colour as its parent. 
        # Case 3: Inserted Node's sibling's far child is black
        #         (if Inserted node is a left child, then the far child would its sibling's right child)
        #         since either this case (& case 4) or case 2 would be performed we can assume atleast one of inserted node's 
        #         sibling's kid is red. Purpose of this case is just to make inserted node's sibling's far child red.
        #         Colour inserted node's sibling's non far child black & the inserted node's sibling red.
        #         Perform a rotation away from the inserted node on inserted node's sibling. Now inserted node's sibling's far child is red.
        # Case 4: Inserted Node's sibling's far child is red 
        #         (if inserted Node's sibling's far child is red then we jump directly to case 4)
        #         Colour inserted node's sibling's colour as same its parent
        #         Colour inserted node's sibling's far child & inserted node's parent black 
        #         perform a rotation on inserted node's parent towards inserted node.
        #         (if inserted node is a right child then perform a right rotate)
        
        while discrepantLeaf is not self.root and discrepantLeaf.colour is black:
            if discrepantLeaf.parent.left is discrepantLeaf:
                sibling_discrepantLeaf = discrepantLeaf.parent.right
                if sibling_discrepantLeaf.colour is red:
                    sibling_discrepantLeaf.colour, discrepantLeaf.parent.colour = black, red
                    self.rotate(discrepantLeaf.parent, left=True)
                    sibling_discrepantLeaf = discrepantLeaf.parent.right
                if sibling_discrepantLeaf.right.colour == sibling_discrepantLeaf.left.colour == black:
                    sibling_discrepantLeaf.colour, discrepantLeaf = red, discrepantLeaf.parent
                else:
                    if sibling_discrepantLeaf.right.colour is black:
                        sibling_discrepantLeaf.left.colour, sibling_discrepantLeaf.colour = black, red
                        self.rotate(sibling_discrepantLeaf, right=True)
                        sibling_discrepantLeaf = discrepantLeaf.parent.right 
                    sibling_discrepantLeaf.colour = discrepantLeaf.parent.colour
                    sibling_discrepantLeaf.right.colour = discrepantLeaf.parent.colour = black 
                    self.rotate(discrepantLeaf.parent, left=True)
                    break
            else:
                sibling_discrepantLeaf = discrepantLeaf.parent.left 
                if sibling_discrepantLeaf is red:
                    sibling_discrepantLeaf.colour, discrepantLeaf.parent.colour = black, red
                    self.rotate(discrepantLeaf.parent, right=True)
                    sibling_discrepantLeaf = discrepantLeaf.parent.left
                if sibling_discrepantLeaf.right == sibling_discrepantLeaf.left == black:
                    sibling_discrepantLeaf.colour, discrepantLeaf = red, discrepantLeaf.parent
                else:
                    if sibling_discrepantLeaf.left is black:
                        sibling_discrepantLeaf.right.colour, sibling_discrepantLeaf.colour = black, red
                        self.rotate(sibling_discrepantLeaf,left=True)
                    sibling_discrepantLeaf.colour = sibling_discrepantLeaf.parent.colour
                    sibling_discrepantLeaf.left.colour = discrepantLeaf.parent.colour = black
                    self.rotate(discrepantLeaf.parent, right=True)
                    break
        discrepantLeaf.colour = black

#since RedBlackTree directly inherits from BinarySearchTree it should be declared just like a binarySearchTree