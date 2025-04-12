import React from "react";
import { TbRepeat } from "react-icons/tb";

import { declineYear } from "@utils/decline";

import Input from "../Input";
import Checkbox from "@components/Checkbox/Checkbox";

import "./preview.css";

const InputPreview: React.FC = () => {
  return (
    <div className="preview-container">
      <Checkbox name="remember" label="Запомнить меня" />

      <Input type="text" name="name" label="Имя" placeholder="Имя" />
      <Input type="text" name="surname" label="Фамилия" />
      <Input type="date" name="birthday" label="Дата рождения" />
      <Input type="email" name="email" label="Эл. почта" />
      <Input
        type="experience"
        name="experience"
        label="Опыт"
        getUnit={declineYear}
      />
      <Input type="number" name="number" label="Число" getUnit={declineYear} />
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
