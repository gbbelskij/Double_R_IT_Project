export interface Course {
  id: number;
  title: string;
  duration: number;
  description: string;
  url: string;
}

export const courses: Course[] = [
  {
    id: 0,
    title: "Введение в JavaScript",
    duration: 10,
    description: "Изучите основы программирования на JavaScript.",
    url: "/courses/javascript-intro",
  },
  {
    id: 1,
    title: "React для начинающих",
    duration: 52,
    description: "Начните работу с React и создайте свое первое приложение.",
    url: "/courses/react-beginners",
  },
  {
    id: 2,
    title: "Продвинутый TypeScript",
    duration: 30,
    description: "Освоите TypeScript для large-scale приложений.",
    url: "/courses/advanced-typescript",
  },
  {
    id: 3,
    title: "CSS Grid и Flexbox",
    duration: 15,
    description: "Изучите современные методы верстки CSS.",
    url: "/courses/css-grid-flexbox",
  },
  {
    id: 4,
    title: "Node.js основы",
    duration: 40,
    description: "Создавайте серверные приложения с помощью Node.js.",
    url: "/courses/nodejs-essentials",
  },
  {
    id: 5,
    title: "Web Accessibility",
    duration: 20,
    description: "Сделайте ваши веб-сайты доступными для всех.",
    url: "/courses/web-accessibility",
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