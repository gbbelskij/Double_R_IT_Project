import { useState } from "react";
import Select, { components } from "react-select";
import makeAnimated from "react-select/animated";
import { MdOutlineWorkOutline } from "react-icons/md";
import CustomOption from "@components/CustomOption/CustomOption"; 

const animatedComponents = makeAnimated();

const Placeholder = (props: any) => {
  // Не отображать плейсхолдер если есть выбранное значение
  if (props.selectProps.value) return null;
  
  return (
    <components.Placeholder {...props}>
      <div style={{ 
        display: "flex",
        alignItems: "center",
        gap: "8px",
        position: "absolute",
        left: 0,
        top: 0,
        right: 0,
        bottom: 0
      }}>
        <MdOutlineWorkOutline size={20} color="white" />
        <span>{props.children}</span>
      </div>
    </components.Placeholder>
  );
};

// Компонент для отображения выбранного значения с иконкой
const SingleValue = ({ children, ...props }: any) => (
  <components.SingleValue {...props}>
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: "8px",
      width: "100%",
      position: "relative",
      zIndex: 2
    }}>
      <MdOutlineWorkOutline size={20} color="white" />
      <span style={{
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
        flex: 1,
        position: "relative"
      }}>
        {children}
      </span>
    </div>
  </components.SingleValue>
);

// Компонент для списка опций с иконкой
const Option = (props: any) => (
  <components.Option {...props}>
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "8px",
      }}
    >
      <MdOutlineWorkOutline size={20} color="white" style={{ flex: 1 }} />
      {props.data.label}
    </div>
  </components.Option>
);

const CustomSelect = () => {
  const [selectedOption, setSelectedOption] = useState<any>(null);

  const options = [
    { value: "frontend", label: "Frontend-разработчик" },
    { value: "backend", label: "Backend-разработчик" },
    { value: "manager", label: "Менеджер" },
    { value: "long-text", label: "Очень длинное название: aaaaaaaaaaaaaaaaaaaa" },
  ];

  // Кастомные стили для select
  const customStyles = {
    container: (provided: any) => ({ 
      ...provided,
      flex: "1",
      display: 'flex',
     }),
     control: (provided: any, state: any) => ({
      ...provided,
      display: "flex",
      alignItems: "center",
      backgroundColor: "transparent",
      border: "2px solid white",
      borderRadius: state.isFocused ? "20px 20px 0 0" : "20px",
      color: "#fff",
      fontSize: "16px",
      padding: "10px 16px",
      width: "500px",
      minHeight: "56px",
      boxShadow: "none",
      position: "relative",
      "&:hover": {
        borderColor: "white"
      }
    }),
    menu: (provided: any) => ({
      ...provided,
      backgroundColor: "none",
      borderRadius: "0 0 20px 20px",
      boxShadow: "0",
      marginTop: "1px",
      border: "3px solid white",
      borderTop: "none",
      width: "500px",
    }),
    option: (provided: any) => ({
      ...provided,
      backgroundColor: "transparent",
      padding: "8px 16px", /* Добавьте отступы */
      ":hover": {
        backgroundColor: "#3a3e45"
      }
    }),
    singleValue: (provided: any) => ({
      ...provided,
      display: "flex !important",
      alignItems: "center",
      margin: 0,
      maxWidth: "100%",
      position: "relative",
      zIndex: 2
    }),
    indicatorSeparator: () => ({
      display: "none",
    }),
    dropdownIndicator: (provided: any) => ({
      ...provided,
      color: "#fff",
    }),
    placeholder: (provided: any) => ({
      ...provided,
      color: "#fff !important",
      margin: 0,
      width: "100%"
    }),
    valueContainer: (provided: any) => ({
      ...provided,
      padding: "0 !important",
      gap: "8px",
      position: "relative",
      overflow: "hidden", // Изменено с visible
      flexWrap: "nowrap"
    }),
    indicatorsContainer: (provided: any) => ({
      ...provided,
      position: "relative",
      zIndex: 3, // Увеличено значение
      paddingLeft: "8px",
      backgroundColor: "transparent"
    }),
  };

  return (
    <div>
      <h2
        style={{
          marginBottom: "10px",
          fontWeight: 700,
          fontSize: "20px",
          width: "500px",
        }}
      >
        Должность
      </h2>
      <Select
        styles={customStyles}
        components={{
          ...animatedComponents, // Сначала анимированные компоненты
          Option: CustomOption,  // Затем переопределяем Option
          SingleValue,
          Placeholder, 
          IndicatorSeparator: () => null
        }}
        options={options}
        value={selectedOption}
        onChange={setSelectedOption}
        placeholder="Выберите должность"
        isSearchable={false}
        theme={(theme) => ({
          ...theme,
          colors: {
            ...theme.colors,
            primary: "#fff",
            primary25: "#3a3e45",
          },
        })}
      />
    </div>
  );
};

export default CustomSelect;