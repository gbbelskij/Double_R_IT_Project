type ButtonTypes = 'text' | 'date' | 'email' | 'number' | 'password' | 'checkbox';

export interface InputProps {
    type: ButtonTypes;
    leftIcon?: React.ReactNode;
    rightIcon?: React.ReactNode;
    name: string;
    label: string;
    hideIcons?: boolean;
    postfix?: string; // надпись для type="number"
    showEye?: boolean; 
    [key: string]: any; 
}