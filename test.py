class Long_substring:
    
    def __init__(self, str_func = None):
        self.input_str = str_func
        self.num = None
        


    def lenlong_substring(self):
        str = self.input_str
        char_index = {}
        left = 0
        max_num = 0
        for right in range(len(str)):
            char = str[right]
            if char in char_index and char_index[char] >=left:
                left = char_index[char]+1
                char_index[char] = right
            
        char_index[char] = right
        
        max_num = max(max_num, right-left+1)
        return max_num
    