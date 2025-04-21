import React from "react";
import { TbRepeat } from "react-icons/tb";
import { MdOutlineWorkOutline } from "react-icons/md";

import { declineYear } from "@utils/decline";

import Input from "../Input";
import Checkbox from "@components/Checkbox/Checkbox";

import "./preview.css";
import Select from "@components/Select/Select";

const options = [
  { value: "frontend", label: "Frontend-разработчик" },
  { value: "backend", label: "Backend-разработчик" },
  { value: "manager", label: "Менеджер" },
  {
    value: "long-text",
    label: "Очень длинное название: aaaaaaaaaaaaaaaaaaaa",
  },
];

const InputPreview: React.FC = () => {
  return (
    <div className="preview-container">
      <Checkbox name="remember" label="Запомнить меня" />

      <Input type="text" name="name" label="Имя" placeholder="Имя" />
      <Input type="text" name="surname" label="Фамилия" placeholder="Фамилия" />
      <Input type="date" name="birthday" label="Дата рождения" />
      <Input type="email" name="email" label="Эл. почта" />
      <Select
        options={options}
        name="post"
        label="Должность"
        placeholder="Выберите должность"
        icon={MdOutlineWorkOutline}
      />
      <Input
        type="experience"
        name="experience"
        label="Опыт"
        getUnit={declineYear}
      />
      <Input
        type="number"
        name="number"
        label="Число"
        placeholder="Введите число"
      />
      <Input type="password" name="password" label="Пароль" />
      <Input
        type="password"
        name="repeat-password"
        label="Повторите пароль"
        icon={TbRepeat}
      />
    </div>
  );
};

export default InputPreview;
