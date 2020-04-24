# static member decorator
def static(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate
# enddef
