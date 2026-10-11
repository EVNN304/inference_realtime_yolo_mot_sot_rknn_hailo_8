from ctypes import LittleEndianStructure, Union
import inspect


class BaseStructure(LittleEndianStructure):
    def __str__(self):
        return self._format_structure()
    
    def __repr__(self):
        return self.__str__()
    
    def _format_structure(self):
        class_name = self.__class__.__name__
        lines = []

        if hasattr(self, "_fields_"):
            for field_definition in self._fields_:
                if len(field_definition) == 3:
                    field_name, field_type, field_bitlen = field_definition
                    value = getattr(self, field_name)
                    formatted_value = f"{field_name}={value}"
                else:
                    field_name, field_type = field_definition
                    value = getattr(self, field_name)
                    formatted_value = self._format_field_value(
                        field_name, field_type, value
                    )
                lines.append(formatted_value)

        return f"{class_name}(" + ", ".join(lines) + ")"
    
    def _format_field_value(self, field_name, field_type, value):
        if self._is_structure_field(field_type):
            nested = value._format_structure()
            return f"{field_name}={nested}"
        
        elif self._is_union_field(field_type):
            return self._format_union(field_name, value)
        
        elif self._is_array_field(field_type):
            return self._format_array(field_name, value)
        
        else:
            return self._format_basic_value(field_name, value)
    
    def _is_structure_field(self, field_type):
        return (inspect.isclass(field_type) and 
                issubclass(field_type, (LittleEndianStructure)) and
                hasattr(field_type, '_fields_'))
    
    def _is_union_field(self, field_type):
        return (inspect.isclass(field_type) and 
                issubclass(field_type, Union))
    
    def _is_array_field(self, field_type):
        return (hasattr(field_type, '_type_') and 
                hasattr(field_type, '_length_'))
    
    def _format_array(self, field_name, array_value):
        array_length = len(array_value)
        elements = [str(array_value[i]) for i in range(array_length)]
        return f"{field_name}=[{', '.join(elements)}]"
            
    def _format_union(self, field_name, union_value):
        lines = []
        if hasattr(union_value, "_fields_"):
            for field_definition in union_value._fields_:
                if len(field_definition) == 3:
                    field_name, field_type, field_bitlen = field_definition
                    value = getattr(self, field_name)
                    formatted_value = f"{field_name}={value}"
                else:
                    field_name, field_type = field_definition
                    value = getattr(union_value, field_name)
                    if isinstance(value, BaseStructure):
                        formatted_value = value._format_structure()
                    else:
                        formatted_value = self._format_basic_value(
                            field_name, value
                        )
                lines.append(formatted_value)
        
        return f"{field_name}=[{','.join(lines)}]"
    
    def _format_basic_value(self, field_name, value):
        if isinstance(value, int):
            return f"{field_name}={value} ({hex(value)})"
        elif isinstance(value, float):
            return f"{value:.4f}"
        else:
            return str(value)