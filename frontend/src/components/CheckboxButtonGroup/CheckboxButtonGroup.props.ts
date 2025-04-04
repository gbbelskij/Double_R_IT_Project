export interface Preference {
  id: number;
  value: string;
}

export interface CheckboxButtonGroupProps {
  preferences: Preference[];
}
