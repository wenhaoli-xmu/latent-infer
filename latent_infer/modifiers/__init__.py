def get_modifier(method: str, model_structure):

    if method == 'origin':
        from .origin import Origin
        return Origin

    elif method == 'greedy':
        from .greedy import Greedy
        return Greedy
    
    elif method == 'pretrain':
        from .pretrain import ModelForTraining
        return ModelForTraining
    
    elif method == 'cotrain':
        from .cotrain import ModelForTraining
        return ModelForTraining

    elif method =='eval':
        from .eval import ModelForEvaluation
        return ModelForEvaluation
