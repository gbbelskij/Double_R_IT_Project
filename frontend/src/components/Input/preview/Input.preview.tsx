import React from "react";
import { TbRepeat } from "react-icons/tb";

import Input from "../Input";
import Checkbox from "@components/Checkbox/Checkbox";

import "./preview.css";

function getYearWord(n: number): string {
  const lastTwoDigits = n % 100;
  const lastDigit = n % 10;

  if (lastTwoDigits > 10 && lastTwoDigits < 20) {
    return "лет";
  }

  if (lastDigit === 1) {
    return "год";
  }

  if (lastDigit >= 2 && lastDigit <= 4) {
    return "года";
  }

  return "лет";
}

const InputPreview: React.FC = () => {
  return (
    <div className="preview-container">
      <Checkbox name="remember" label="Запомнить меня" />

      <Input type="text" name="name" label="Имя" placeholder="Имя" />
      <Input type="text" name="surname" label="Фамилия" />
      <Input type="date" name="birthday" label="Дата рождения" />
      <Input type="email" name="email" label="Эл. почта" />
      <Input type="number" name="number" label="Опыт" getUnit={getYearWord} />
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
