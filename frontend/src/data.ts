export interface Course {
  id: number;
  title: string;
  duration: string; // Меняем на string, чтобы соответствовать CourseProps
  description: string;
  url: string;
  imageSrc: string; // Добавляем imageSrc
}

export const courses: Course[] = [
  {
    id: 0,
    title: "Python-разработчик",
    duration: "10", 
    description: "Вы освоите самый востребованный язык программирования, на котором пишут сайты...",
    url: "/courses/javascript-intro",
    imageSrc: "/images/1.png",
  },
  {
    id: 1,
    title: "Бухгалтер",
    duration: "10",
    description: "Вы научитесь вести бухучёт, работать в 1С, готовить налоговую отчётность...",
    url: "/courses/react-beginners",
    imageSrc: "/images/2.png",
  },
  {
    id: 2,
    title: "Графический дизайнер",
    duration: "10",
    description: "Вы научитесь создавать айдентику для брендов и освоите популярные графические редакторы – от Illustrator до Figma. Сможете зарабатывать уже во время обучения ",
    url: "/courses/advanced-typescript",
    imageSrc: "/images/3.png",
  },
  {
    id: 3,
    title: "CSS Grid и Flexbox",
    duration: "15",
    description: "Изучите современные методы верстки CSS.",
    url: "/courses/css-grid-flexbox",
    imageSrc: "/images/1.png",
  },
  {
    id: 4,
    title: "Node.js основы",
    duration: "40",
    description: "Создавайте серверные приложения с помощью Node.js.",
    url: "/courses/nodejs-essentials",
    imageSrc: "/images/2.png",
  },
  {
    id: 5,
    title: "Web Accessibility",
    duration: "20",
    description: "Сделайте ваши веб-сайты доступными для всех.",
    url: "/courses/web-accessibility",
    imageSrc: "/images/3.png",
  },
];

// Added Preference interface and preferences array
export interface Preference {
  id: number;
  value: string;
}

export const preferences: Preference[] = [
  {
    id: 1,
    value: "Создать пользовательский интерфейс",
  },
  {
    id: 2,
    value: "Реализовать серверную логику для приложения",
  },
  {
    id: 3,
    value: "Проектировать и оптимизировать базы данных",
  },
];