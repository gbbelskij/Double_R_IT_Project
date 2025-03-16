type InputTypes = 'text' | 'date' | 'email' | 'number' | 'password' | 'checkbox';

export interface InputProps {
    type: InputTypes;
    leftIcon?: React.ReactNode;
    rightIcon?: React.ReactNode;
    name: string;
    label: string;
    hideIcons?: boolean;
    postfix?: string; 
    showEye?: boolean;
    defaultValue?: string; 
    unit?: string;
    placeholder?: string;
    [key: string]: any; 
}