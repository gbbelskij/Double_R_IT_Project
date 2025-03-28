export interface Preference {
    id: number;
    value: string;
  }
  
  export interface CheckboxButtonGroupProps {
    preferences: Preference[];
  }


  export interface CourseProps {
    title: string;
    duration: string;
    description: string;
    url: string;
    imageSrc: string; /* Новое свойство для изображения */
  }