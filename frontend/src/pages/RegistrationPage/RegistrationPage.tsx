import Logo from "@components/Logo/Logo";
import Main from "@components/Main/Main";
import { CheckboxButtonGroup } from "@components/CheckboxButtonGroup/CheckboxButtonGroup"; // Проверь путь
import { preferences } from "../../data"; // Проверь путь

import "./RegistrationPage.css";

const RegistrationPage: React.FC = () => {
  return (
    <Main disableHeaderOffset>
      <Logo hasText />
      <h2>Шаг 2: Выберите предпочтения</h2>
      <CheckboxButtonGroup preferences={preferences} />
    </Main>
  );
};

export default RegistrationPage;