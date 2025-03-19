import { InputTypes } from './Input.types';

export interface InputProps {
    type: InputTypes;
    leftIcon?: React.ReactNode;
    name: string;
    label: string;
    hideIcons?: boolean;
    postfix?: string; 
    defaultValue?: string; 
    unit?: string;
    [key: string]: any; 
}