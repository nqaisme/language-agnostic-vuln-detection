from .utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)


def DFG_python(root_node,index_to_code,states):
    assignment=['assignment','augmented_assignment','for_in_clause']
    if_statement=['if_statement']
    for_statement=['for_statement']
    while_statement=['while_statement']
    do_first_statement=['for_in_clause'] 
    def_statement=['default_parameter']
    states=states.copy() 
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':        
        idx,code=index_to_code[(root_node.start_point,root_node.end_point)]
        if root_node.type==code:
            return [],states
        elif code in states:
            return [(code,idx,'comesFrom',[code],states[code].copy())],states
        else:
            if root_node.type=='identifier':
                states[code]=[idx]
            return [(code,idx,'comesFrom',[],[])],states
    elif root_node.type in def_statement:
        name=root_node.child_by_field_name('name')
        value=root_node.child_by_field_name('value')
        DFG=[]
        if value is None:
            indexs=tree_to_variable_index(name,index_to_code)
            for index in indexs:
                idx,code=index_to_code[index]
                DFG.append((code,idx,'comesFrom',[],[]))
                states[code]=[idx]
            return sorted(DFG,key=lambda x:x[1]),states
        else:
            name_indexs=tree_to_variable_index(name,index_to_code)
            value_indexs=tree_to_variable_index(value,index_to_code)
            temp,states=DFG_python(value,index_to_code,states)
            DFG+=temp            
            for index1 in name_indexs:
                idx1,code1=index_to_code[index1]
                for index2 in value_indexs:
                    idx2,code2=index_to_code[index2]
                    DFG.append((code1,idx1,'comesFrom',[code2],[idx2]))
                states[code1]=[idx1]   
            return sorted(DFG,key=lambda x:x[1]),states        
    elif root_node.type in assignment:
        if root_node.type=='for_in_clause':
            right_nodes=[root_node.children[-1]]
            left_nodes=[root_node.child_by_field_name('left')]
        else:
            if root_node.child_by_field_name('right') is None:
                return [],states
            left_nodes=[x for x in root_node.child_by_field_name('left').children if x.type!=',']
            right_nodes=[x for x in root_node.child_by_field_name('right').children if x.type!=',']
            if len(right_nodes)!=len(left_nodes):
                left_nodes=[root_node.child_by_field_name('left')]
                right_nodes=[root_node.child_by_field_name('right')]
            if len(left_nodes)==0:
                left_nodes=[root_node.child_by_field_name('left')]
            if len(right_nodes)==0:
                right_nodes=[root_node.child_by_field_name('right')]
        DFG=[]
        for node in right_nodes:
            temp,states=DFG_python(node,index_to_code,states)
            DFG+=temp
            
        for left_node,right_node in zip(left_nodes,right_nodes):
            left_tokens_index=tree_to_variable_index(left_node,index_to_code)
            right_tokens_index=tree_to_variable_index(right_node,index_to_code)
            temp=[]
            for token1_index in left_tokens_index:
                idx1,code1=index_to_code[token1_index]
                temp.append((code1,idx1,'computedFrom',[index_to_code[x][1] for x in right_tokens_index],
                             [index_to_code[x][0] for x in right_tokens_index]))
                states[code1]=[idx1]
            DFG+=temp        
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in if_statement:
        DFG=[]
        current_states=states.copy()
        others_states=[]
        tag=False
        if 'else' in root_node.type:
            tag=True
        for child in root_node.children:
            if 'else' in child.type:
                tag=True
            if child.type not in ['elif_clause','else_clause']:
                temp,current_states=DFG_python(child,index_to_code,current_states)
                DFG+=temp
            else:
                temp,new_states=DFG_python(child,index_to_code,states)
                DFG+=temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states
    elif root_node.type in for_statement:
        DFG=[]
        for i in range(2):
            right_nodes=[x for x in root_node.child_by_field_name('right').children if x.type!=',']
            left_nodes=[x for x in root_node.child_by_field_name('left').children if x.type!=',']
            if len(right_nodes)!=len(left_nodes):
                left_nodes=[root_node.child_by_field_name('left')]
                right_nodes=[root_node.child_by_field_name('right')]
            if len(left_nodes)==0:
                left_nodes=[root_node.child_by_field_name('left')]
            if len(right_nodes)==0:
                right_nodes=[root_node.child_by_field_name('right')]
            for node in right_nodes:
                temp,states=DFG_python(node,index_to_code,states)
                DFG+=temp
            for left_node,right_node in zip(left_nodes,right_nodes):
                left_tokens_index=tree_to_variable_index(left_node,index_to_code)
                right_tokens_index=tree_to_variable_index(right_node,index_to_code)
                temp=[]
                for token1_index in left_tokens_index:
                    idx1,code1=index_to_code[token1_index]
                    temp.append((code1,idx1,'computedFrom',[index_to_code[x][1] for x in right_tokens_index],
                                 [index_to_code[x][0] for x in right_tokens_index]))
                    states[code1]=[idx1]
                DFG+=temp   
            if  root_node.children[-1].type=="block":
                temp,states=DFG_python(root_node.children[-1],index_to_code,states)
                DFG+=temp 
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in while_statement:  
        DFG=[]
        for i in range(2):
            for child in root_node.children:
                temp,states=DFG_python(child,index_to_code,states)
                DFG+=temp    
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states        
    else:
        DFG=[]
        for child in root_node.children:
            if child.type in do_first_statement:
                temp,states=DFG_python(child,index_to_code,states)
                DFG+=temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp,states=DFG_python(child,index_to_code,states)
                DFG+=temp
        
        return sorted(DFG,key=lambda x:x[1]),states



def DFG_c_cpp(root_node, index_to_code, states):
    assignment = ['assignment_expression', 'update_expression', 'init_declarator']
    if_statement = ['if_statement']
    for_statement = ['for_statement', 'range_based_for_statement']
    while_statement = ['while_statement', 'do_statement']
    do_first_statement = []
    def_statement = ['parameter_declaration', 'optional_parameter_declaration']
    
    states = states.copy()
    
    if (len(root_node.children) == 0 or root_node.type in ['string_literal', 'char_literal']) and root_node.type != 'comment':        
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states
            
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('declarator')
        value = root_node.child_by_field_name('default_value') or root_node.child_by_field_name('value')
        DFG = []
        if value is None:
            if name is not None:
                indexs = tree_to_variable_index(name, index_to_code)
                for index in indexs:
                    idx, code = index_to_code[index]
                    DFG.append((code, idx, 'comesFrom', [], []))
                    states[code] = [idx]
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code) if name else []
            value_indexs = tree_to_variable_index(value, index_to_code) if value else []
            temp, states = DFG_c_cpp(value, index_to_code, states)
            DFG += temp            
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                states[code1] = [idx1]   
            return sorted(DFG, key=lambda x: x[1]), states  
                  
    elif root_node.type in assignment:
        left_nodes = []
        right_nodes = []
        if root_node.type == 'assignment_expression':
            left_nodes = [root_node.child_by_field_name('left')]
            right_nodes = [root_node.child_by_field_name('right')]
        elif root_node.type == 'init_declarator':
            left_nodes = [root_node.child_by_field_name('declarator')]
            right_nodes = [root_node.child_by_field_name('value')]
        elif root_node.type == 'update_expression':
            argument = root_node.child_by_field_name('argument')
            if argument is None and len(root_node.children) > 0:
                argument = root_node.children[0] if root_node.children[0].type not in ['++', '--'] else root_node.children[-1]
            left_nodes = [argument]
            right_nodes = [argument]

        left_nodes = [x for x in left_nodes if x is not None and x.type != ',']
        right_nodes = [x for x in right_nodes if x is not None and x.type != ',']
        
        DFG = []
        for node in right_nodes:
            temp, states = DFG_c_cpp(node, index_to_code, states)
            DFG += temp
            
        for left_node, right_node in zip(left_nodes, right_nodes):
            left_tokens_index = tree_to_variable_index(left_node, index_to_code)
            right_tokens_index = tree_to_variable_index(right_node, index_to_code)
            temp = []
            for token1_index in left_tokens_index:
                idx1, code1 = index_to_code[token1_index]
                temp.append((code1, idx1, 'computedFrom', 
                             [index_to_code[x][1] for x in right_tokens_index],
                             [index_to_code[x][0] for x in right_tokens_index]))
                states[code1] = [idx1]
            DFG += temp        
        return sorted(DFG, key=lambda x: x[1]), states
        
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        tag = False
        
        if root_node.child_by_field_name('alternative') is not None:
            tag = True
            
        for child in root_node.children:
            if child.type == 'else':
                continue
            if child == root_node.child_by_field_name('alternative'):
                temp, new_states = DFG_c_cpp(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
            else:
                temp, current_states = DFG_c_cpp(child, index_to_code, current_states)
                DFG += temp
                
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
            
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
        
    elif root_node.type in for_statement or root_node.type in while_statement:
        DFG = []
        for i in range(2):
            if root_node.type == 'range_based_for_statement':
                left_node = root_node.child_by_field_name('left')
                right_node = root_node.child_by_field_name('right')
                body_node = root_node.child_by_field_name('body')
                
                if right_node:
                    temp, states = DFG_c_cpp(right_node, index_to_code, states)
                    DFG += temp
                if left_node and right_node:
                    left_tokens_index = tree_to_variable_index(left_node, index_to_code)
                    right_tokens_index = tree_to_variable_index(right_node, index_to_code)
                    temp_dfg = []
                    for token1_index in left_tokens_index:
                        idx1, code1 = index_to_code[token1_index]
                        temp_dfg.append((code1, idx1, 'computedFrom',
                                     [index_to_code[x][1] for x in right_tokens_index],
                                     [index_to_code[x][0] for x in right_tokens_index]))
                        states[code1] = [idx1]
                    DFG += temp_dfg
                if body_node:
                    temp, states = DFG_c_cpp(body_node, index_to_code, states)
                    DFG += temp
            else:
                for child in root_node.children:
                    temp, states = DFG_c_cpp(child, index_to_code, states)
                    DFG += temp    
                    
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states        
        
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_c_cpp(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_c_cpp(child, index_to_code, states)
                DFG += temp
        
        return sorted(DFG, key=lambda x: x[1]), states